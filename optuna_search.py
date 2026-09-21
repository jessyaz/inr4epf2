import os
import sys
from pathlib import Path
import numpy as np
import optuna
import torch
import yaml
from omegaconf import OmegaConf, open_dict

from naming import experiment_name, main_run
from runner import build_loaders, build_model
from utils.mlflow_logger import MLflowLogger
from utils.trainer import train

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_FILE_MAP = {
    "masked_transformer": "transformer_masked.yaml",
    "imputed_transformer": "transformer_imputed.yaml",
    "dnn": "dnn.yaml",
}

ISO_1M_CONFIGS = {
    "masked_transformer": {
        "model": {
            "embedding_dim": 256,
            "num_heads": 8,
            "dim_feedforward": 512,
            "num_layers": 6,
            "normalize_first": True,
            "activation": "gelu",
            "use_lookback": True,
        }
    },
    "imputed_transformer": {
        "model": {
            "embedding_dim": 256,
            "num_heads": 8,
            "dim_feedforward": 512,
            "num_layers": 6,
            "normalize_first": True,
            "activation": "gelu",
            "use_lookback": True,
        }
    },
    "dnn": {
        "model": {
            "hidden_dim": 512,
            "num_layers": 5,
            "use_lookback": True,
        }
    },
}


def run_single_experiment(cfg):
    """Exécute l'entraînement complet via train() et logge dans MLflow."""
    import uuid

    with open_dict(cfg):
        cfg.model_uid = uuid.uuid4().hex[:8]
        if "mlflow" not in cfg or cfg.mlflow is None:
            cfg.mlflow = {}

        cfg.mlflow.experiment_name = f"optuna_{cfg.registry}"

        try:
            cfg.mlflow.run_name = main_run(cfg.registry, cfg.model_uid)
        except Exception:
            cfg.mlflow.run_name = f"run_{cfg.registry}_{cfg.model_uid}"

        cfg.run_dir = (Path("runs") / cfg.model_uid).as_posix()

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else cfg.device
    )

    train_loader, val_loader, _, _ = build_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = model.configure_optimizer()

    loaders = {"train_loader": train_loader, "val_loader": val_loader}

    with MLflowLogger(cfg) as logger:
        if optimizer is None:
            results = model.fit(loaders, device, logger)
        else:
            results = train(model, loaders, optimizer, device, logger)

        ckpt = run_dir / "model.pth"
        model.save(ckpt)
        logger.log_checkpoint(str(ckpt))

    return results["val_loss"]["MSE"]


def optimize_and_evaluate(registry_name, n_trials=20, seeds=[0, 1, 2, 3, 4]):
    print(f"\n==================================================")
    print(f"  LANCEMENT OPTUNA : {registry_name.upper()} (~1M Params)")
    print(f"==================================================")

    yaml_filename = MODEL_FILE_MAP.get(registry_name, f"{registry_name}.yaml")
    config_path = Path("conf") / yaml_filename

    if not config_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {config_path}")

    base_cfg = OmegaConf.load(config_path)

    if registry_name in ISO_1M_CONFIGS:
        override_cfg = OmegaConf.create(ISO_1M_CONFIGS[registry_name])
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    with open_dict(base_cfg):
        base_cfg.registry = registry_name
        if "masking" not in base_cfg:
            base_cfg.masking = {}
        base_cfg.masking.rate = 0.0

    def objective(trial):
        cfg = base_cfg.copy()

        lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.35, step=0.05)
        stride_train = trial.suggest_categorical("stride_train", [1, 2, 3, 6])

        with open_dict(cfg):
            if "optim" not in cfg.model:
                cfg.model.optim = {}
            cfg.model.optim.lr = lr
            cfg.model.optim.weight_decay = wd
            cfg.model.dropout = dropout
            if "window" not in cfg:
                cfg.window = {}
            cfg.window.stride_train = stride_train
            cfg.seed = 0

        try:
            return run_single_experiment(cfg)
        except Exception as e:
            print(f"[Trial Failed - {registry_name}] : {e}")
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"\n[+] Meilleurs hyperparamètres pour {registry_name} à r=0.0 :")
    print(best_params)

    best_config_dict = {
        "registry": registry_name,
        "model": {
            **ISO_1M_CONFIGS[registry_name]["model"],
            "dropout": best_params["dropout"],
            "optim": {
                "lr": best_params["lr"],
                "weight_decay": best_params["weight_decay"],
            },
        },
        "window": {"stride_train": best_params["stride_train"]},
    }

    yaml_output_path = Path("conf") / f"best_{registry_name}.yaml"
    with open(yaml_output_path, "w") as f:
        yaml.dump(best_config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"[+] Configuration enregistrée dans : {yaml_output_path}")

    print(f"\n--- Évaluation finale sur 5 Seeds (seeds={seeds}) ---")
    val_mses = []

    final_cfg = base_cfg.copy()
    with open_dict(final_cfg):
        if "optim" not in final_cfg.model:
            final_cfg.model.optim = {}
        final_cfg.model.optim.lr = best_params["lr"]
        final_cfg.model.optim.weight_decay = best_params["weight_decay"]
        final_cfg.model.dropout = best_params["dropout"]
        final_cfg.window.stride_train = best_params["stride_train"]

    for seed in seeds:
        cfg_seed = final_cfg.copy()
        with open_dict(cfg_seed):
            cfg_seed.seed = seed

        val_mse = run_single_experiment(cfg_seed)
        val_mses.append(val_mse)
        print(f"  Seed {seed} | Val MSE: {val_mse:.6f}")

    print(f"\n[RÉSULTAT FINAL {registry_name.upper()}]")
    print(
        f"MSE Moyenne Validation : {np.mean(val_mses):.6f} +/- {np.std(val_mses):.6f}"
    )


if __name__ == "__main__":
    models_to_run = ["masked_transformer", "imputed_transformer", "dnn"]

    for model_name in models_to_run:
        optimize_and_evaluate(
            model_name, n_trials=20, seeds=[0, 1, 2, 3, 4]
        )