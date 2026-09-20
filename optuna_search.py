import sys
import os
from pathlib import Path
import optuna
from omegaconf import OmegaConf, open_dict
import numpy as np
import torch
import yaml
from hydra import initialize, compose

from naming import main_run, experiment_name
from utils.mlflow_logger import MLflowLogger
from utils.tester import test
from utils.trainer import train
from runner import build_model, build_loaders

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configurations verrouillées à ~1M de paramètres par architecture
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
    }
}


def run_single_experiment(cfg):
    """ Exécute un entraînement complet basé sur la logique du runner (avec MLflow) """
    import uuid

    ablated = not cfg.model.get("use_lookback", True)

    with open_dict(cfg):
        cfg.model_uid = uuid.uuid4().hex[:8]
        cfg.mlflow.experiment_name = experiment_name(cfg)
        cfg.mlflow.run_name = main_run(cfg.registry, cfg.model_uid)
        cfg.run_dir = (Path("runs") / cfg.model_uid).as_posix()

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device

    train_loader, val_loader, test_loader, scaler = build_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = model.configure_optimizer()

    loaders = {"train_loader": train_loader, "val_loader": val_loader}

    with MLflowLogger(cfg) as logger:
        if optimizer is None:
            model.fit(loaders, device, logger)
        else:
            train(model, loaders, optimizer, device, logger)

        if hasattr(model, "prepare_test"):
            model.prepare_test(test_loader, logger)

        ckpt = run_dir / "model.pth"
        model.save(ckpt)
        logger.log_checkpoint(str(ckpt))

        results = test(model, test_loader, scaler, device, logger)
        logger.tester_flag = True

    return results["val_loss"]["MAE"], results["test_loss"]["MAE"]


def optimize_and_evaluate(registry_name, n_trials=20, seeds=[0, 1, 2, 3, 4]):
    print(f"\n==================================================")
    print(f"  LANCEMENT OPTUNA : {registry_name.upper()} (~1M Params)")
    print(f"==================================================")

    # Charger la configuration via l'API Hydra
    with initialize(version_base=None, config_path="conf"):
        base_cfg = compose(config_name="config" if os.path.exists("conf/config.yaml") else "main" if os.path.exists("conf/main.yaml") else None,
                           overrides=[f"registry={registry_name}"])

    # Injecter l'architecture 1M et forcer r=0.0
    with open_dict(base_cfg):
        base_cfg.registry = registry_name

        if registry_name in ISO_1M_CONFIGS:
            for k, v in ISO_1M_CONFIGS[registry_name]["model"].items():
                base_cfg.model[k] = v

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
            cfg.window.stride_train = stride_train
            cfg.seed = 0

        try:
            val_mae, _ = run_single_experiment(cfg)
            return val_mae
        except Exception as e:
            print(f"[Trial Failed - {registry_name}] : {e}")
            return float("inf")

    # Recherche Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"\n[+] Meilleurs hyperparamètres pour {registry_name} à r=0.0 :")
    print(best_params)

    # Sauvegarde YAML
    best_config_dict = {
        "registry": registry_name,
        "model": {
            **ISO_1M_CONFIGS[registry_name]["model"],
            "dropout": best_params["dropout"],
            "optim": {
                "lr": best_params["lr"],
                "weight_decay": best_params["weight_decay"]
            }
        },
        "window": {
            "stride_train": best_params["stride_train"]
        }
    }

    Path("conf").mkdir(exist_ok=True)
    yaml_output_path = Path("conf") / f"best_{registry_name}.yaml"
    with open(yaml_output_path, "w") as f:
        yaml.dump(best_config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"[+] Configuration optimale enregistrée dans : {yaml_output_path}")

    # Evaluation 5 seeds
    print(f"\n--- Évaluation finale sur 5 Seeds (seeds={seeds}) ---")
    val_maes, test_maes = [], []

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

        val_mae, test_mae = run_single_experiment(cfg_seed)
        val_maes.append(val_mae)
        test_maes.append(test_mae)
        print(f"  Seed {seed} | Val MAE: {val_mae:.4f} | Test MAE: {test_mae:.4f}")

    print(f"\n[RÉSULTAT FINAL {registry_name.upper()}]")
    print(f"MAE Moyenne Validation : {np.mean(val_maes):.4f} +/- {np.std(val_maes):.4f}")
    print(f"MAE Moyenne Test       : {np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f}")


if __name__ == "__main__":
    models_to_run = ["masked_transformer", "imputed_transformer", "dnn"]

    for model_name in models_to_run:
        optimize_and_evaluate(model_name, n_trials=20, seeds=[0, 1, 2, 3, 4])