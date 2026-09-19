import uuid
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, open_dict

from datasets.loader import build_loader, load_market
from naming import ablated_run, experiment_name, main_run
from utils.mlflow_logger import MLflowLogger
from utils.tester import test
from utils.trainer import train

from models.baselines.mlp import Model as mlp
from models.baselines.lear import Model as lear
from models.baselines.dnn import Model as dnn
from models.inr import Model as inr
from models.git_interfaces.epf_transformer import Model as epf_transformer
#from models.masked_transformer import Model as masked_transformer
from models.transformer import Model as transformer

MODEL_REGISTRY = {
    "mlp": mlp,
    "lear": lear,
    "dnn": dnn,
    "inr": inr,
    "epf_transformer": epf_transformer,
   # "masked_transformer": masked_transformer,
    "masked_transformer": transformer,
    "imputed_transformer": transformer,
}


def build_model(cfg):
    if cfg.registry not in MODEL_REGISTRY:
        raise ValueError(
            f"registry inconnu : '{cfg.registry}'. "
            f"Disponibles : {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[cfg.registry](cfg)


def build_loaders(cfg):
    d = load_market(cfg.dataset.name, cfg.dataset.processed_dir)
    w, m = cfg.window, cfg.masking

    common = dict(lookback=w.lookback, horizon=w.horizon,
                  batch_size=cfg.dataset.batch_size,
                  num_workers=cfg.dataset.num_workers)

    train_loader = build_loader(d, "train", stride=w.stride_train,
                                rate=0.0, shuffle=True, **common)
    val_loader = build_loader(d, "val", stride=w.stride_eval,
                              rate=0.0, **common)
    test_loader = build_loader(d, "test", stride=w.stride_eval,
                               rate=m.rate, mechanism=m.mechanism,
                               block_mean=m.block_mean, seed=cfg.seed,
                               **common)

    return train_loader, val_loader, test_loader, d["scaler"]


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    ablated = not cfg.model.get("use_lookback", True)

    with open_dict(cfg):
        cfg.model_uid = uuid.uuid4().hex[:8]
        cfg.mlflow.experiment_name = experiment_name(cfg)
        cfg.mlflow.run_name = (ablated_run(cfg.registry, cfg.model_uid)
                               if ablated
                               else main_run(cfg.registry, cfg.model_uid))
        cfg.run_dir = (Path("runs") / cfg.model_uid).as_posix()

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else cfg.device)
    print(f"[{cfg.mlflow.run_name}] {cfg.mlflow.experiment_name} "
          f"| seed={cfg.seed} | device={device}")

    train_loader, val_loader, test_loader, scaler = build_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = model.configure_optimizer()

    loaders = {"train_loader": train_loader, "val_loader": val_loader}

    with MLflowLogger(cfg) as logger:

        if optimizer is None:
            model.fit(loaders, device, logger)       # forme close (LEAR...)
        else:
            train(model, loaders, optimizer, device, logger)

        if hasattr(model, "prepare_test"):
            model.prepare_test(test_loader, logger)

        ckpt = run_dir / "model.pth"
        model.save(ckpt)                             # format propre au modele
        logger.log_checkpoint(str(ckpt))

        print(f"test (rate={cfg.masking.rate}, {cfg.masking.mechanism}) :")
        results = test(model, test_loader, scaler, device, logger)
        logger.tester_flag = True

    print(f"\nmodel_uid = {cfg.model_uid}")
    return results["test_loss"]["MAE"]


if __name__ == "__main__":
    main()