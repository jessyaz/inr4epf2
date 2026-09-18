"""POD -- Partial Observability Degradation.

Evalue la degradation d'un modele deja estime quand une part croissante de la
fenetre de lookback devient inobservable.

Ce script N'ENTRAINE RIEN. Il recupere dans MLflow les runs principaux
`{registry}_main_{uid}` de l'experience `icassp_{dataset}_{registry}`,
produits par runner.py, charge leur checkpoint, et balaie la grille en
inference seule. C'est ce qui isole l'effet mesure : la degradation vient de
l'observabilite a l'inference, pas d'un reapprentissage sur donnees degradees.

Les points de grille sont enregistres comme SOUS-RUNS du run principal
lui-meme, qui est rouvert le temps du balayage : un seul run parent par
modele estime, sa courbe accrochee dessous.

Un modele a gradient a plusieurs runs principaux (un par graine
d'entrainement) ; chacun donne sa propre courbe. Un modele deterministe n'en
a qu'un, et seules les graines de masque font varier ses resultats.

Deux notions d'ablation, distinctes :
  - rate = 1.0        le modele estime AVEC lookback, prive de toute
                      observation a l'inference. Asymptote du balayage.
  - run `_ablated_`   le modele reestime SANS lookback (runner.py avec
                      model.use_lookback=false). Mieux specifie pour la tache
                      sans historique, donc borne inferieure plus juste.
                      Recupere ici s'il existe, jamais estime.

Usage :
    uv run partial_observability_degradation.py --config-name=lear dataset.name=DE
    uv run partial_observability_degradation.py --config-name=mlp  dataset.name=DE \\
        pod.model_uid=a3f1b2c9        # restreint a un seul run principal
"""

import itertools
import json
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from datasets.loader import build_loader, load_market
from naming import ABLATED, MAIN, child_run, experiment_name, parse_uid
from utils.mlflow_logger import download_checkpoint, find_runs
from utils.tester import test

from runner import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# recuperation des modeles estimes
# ---------------------------------------------------------------------------

def locate(cfg, kind=MAIN):
    """[(client, run, uid)] des runs `{registry}_{kind}_*` de l'experience.

    Restreint a pod.model_uid s'il est renseigne. La recherche et le repli
    serveur -> store local sont assures par find_runs (utils.mlflow_logger).
    """
    reg = cfg.registry
    want = cfg.pod.get("model_uid", None)

    out = []
    for client, run, name in find_runs(experiment_name(cfg), f"{reg}_{kind}_"):
        uid = parse_uid(name, reg)
        if want and uid != want:
            continue
        out.append((client, run, uid))
    return out


def load_model(cfg, client, run, ckpt_dir, device):
    path = download_checkpoint(client, run, dest=str(ckpt_dir))
    return MODEL_REGISTRY[cfg.registry].load(cfg, path).to(device)


def ensure_ready(model, cfg, d, common, seed):
    """Prepare un modele a recalibration glissante s'il ne l'est pas deja.

    Un checkpoint porte deja ses calibrations (_recal) : les refaire serait
    du calcul perdu. On ne les reconstruit que si elles manquent, ou si la
    calibration doit elle aussi subir la degradation.
    """
    if not hasattr(model, "prepare_test"):
        return
    if cfg.pod.recalib_on_masked:
        return                       # refait a chaque point, dans evaluate()
    if getattr(model, "_recal", None):
        return                       # deja dans le checkpoint

    if getattr(model, "_train_X", None) is None:
        print("      [POD] calibrations absentes du checkpoint et historique "
              "non serialise : evaluation sans recalibration glissante")
        return

    clean = build_loader(d["test"], stride=cfg.window.stride_eval,
                         rate=0.0, seed=seed, **common)
    model.prepare_test(clean, None)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def build_common(cfg):
    w = cfg.window
    return dict(lookback=w.lookback, horizon=w.horizon,
                batch_size=cfg.dataset.batch_size,
                num_workers=cfg.dataset.num_workers)


def evaluate(model, cfg, d, common, rate, mechanism, seed, device):
    w, m = cfg.window, cfg.masking
    loader = build_loader(d["test"], stride=w.stride_eval,
                          rate=rate, mechanism=mechanism,
                          block_mean=m.block_mean, seed=seed, **common)

    if hasattr(model, "prepare_test") and cfg.pod.recalib_on_masked:
        model.prepare_test(loader)

    return test(model, loader, d["scaler"], device, logger=None)["test_loss"]


def log_child(name, params, metrics):
    """Sous-run imbrique ; un echec n'interrompt jamais le balayage."""
    try:
        with mlflow.start_run(nested=True, run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception as e:
        print(f"      [mlflow] sous-run '{name}' non enregistre : {e}")


def grid_points(cfg):
    """(taux, mecanisme, graine) a evaluer.

    A rate = 1.0 plus rien n'est observe : le mecanisme et la graine sont
    sans effet, un seul point suffit au lieu de |mech| x |seeds| identiques.
    """
    rates = sorted(set(float(r) for r in cfg.pod.rates))
    mechs = list(cfg.pod.mechanisms)
    seeds = list(cfg.pod.mask_seeds)

    pts = [(r, m, s) for r, m, s in itertools.product(rates, mechs, seeds)
           if 0 < r < 1.0]
    if 1.0 in rates:
        pts.append((1.0, mechs[0], seeds[0]))
    return pts


# ---------------------------------------------------------------------------
# une courbe, accrochee sous son run principal
# ---------------------------------------------------------------------------

def sweep_one(cfg, d, device, client, run, uid, rows, ckpt_dir):
    reg, market = cfg.registry, cfg.dataset.name
    mechs = list(cfg.pod.mechanisms)
    seeds = list(cfg.pod.mask_seeds)
    common = build_common(cfg)

    print(f"\n[POD] {reg}_{MAIN}_{uid} | {market}")
    model = load_model(cfg, client, run, ckpt_dir, device)
    ensure_ready(model, cfg, d, common, seeds[0])

    base = dict(market=market, model=reg, uid=uid, variant="full")
    pts = grid_points(cfg)

    # on rouvre le run principal : la courbe s'accroche dessous plutot que
    # sous un nouveau parent
    mlflow.set_tracking_uri(client.tracking_uri)
    try:
        parent = mlflow.start_run(run_id=run.info.run_id)
    except Exception as e:
        print(f"      [mlflow] run principal non rouvert ({e}) ; "
              f"les resultats restent ecrits dans le CSV")
        parent = None

    try:
        # le point de reference est re-mesure ici, par le meme chemin de code
        # que le reste de la courbe : un checkpoint mal recharge se verrait
        ref = evaluate(model, cfg, d, common, 0.0, mechs[0], seeds[0], device)
        print(f"  reference  MAE {ref['MAE']:.4f}  rMAE {ref['rMAE']:.4f}")
        rows.append(dict(**base, rate=0.0, mechanism="none",
                         mask_seed=seeds[0], **ref))

        for k, (rate, mech, ms) in enumerate(pts, 1):
            res = evaluate(model, cfg, d, common, rate, mech, ms, device)
            rows.append(dict(**base, rate=rate, mechanism=mech,
                             mask_seed=ms, **res))
            print(f"  [{k:3d}/{len(pts)}] r={rate:<5} {mech:<6} s={ms}  "
                  f"MAE {res['MAE']:.4f}  rMAE {res['rMAE']:.4f}  "
                  f"spike {res['MAE_spike']:.4f}")
            if parent is not None:
                log_child(child_run(reg, uid, rate),
                          dict(rate=rate, mechanism=mech, mask_seed=ms,
                               market=market, registry=reg, variant="full"),
                          res)
    finally:
        if parent is not None:
            try:
                mlflow.end_run("FINISHED")
            except Exception:
                pass


def sweep_ablated(cfg, d, device, rows, ckpt_dir):
    """Evalue les runs `_ablated_` s'il en existe. N'en estime aucun."""
    found = locate(cfg, kind=ABLATED)
    if not found:
        print(f"\n[POD] aucun run '{cfg.registry}_{ABLATED}_*' "
              f"(le produire avec runner.py model.use_lookback=false)")
        return

    common = build_common(cfg)
    mech0 = list(cfg.pod.mechanisms)[0]
    seed0 = list(cfg.pod.mask_seeds)[0]

    for client, run, uid in found:
        model = load_model(cfg, client, run, ckpt_dir, device)
        ensure_ready(model, cfg, d, common, seed0)
        res = evaluate(model, cfg, d, common, 1.0, mech0, seed0, device)
        rows.append(dict(market=cfg.dataset.name, model=cfg.registry, uid=uid,
                         variant="ablated", rate=1.0, mechanism="ablated",
                         mask_seed=seed0, **res))
        print(f"[POD] ablate {uid}  MAE {res['MAE']:.4f}  "
              f"rMAE {res['rMAE']:.4f}")


# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else cfg.device)
    market, reg = cfg.dataset.name, cfg.registry
    exp = experiment_name(cfg)

    print(f"[POD] experiment {exp} | device={device}")
    print(f"      taux        {list(cfg.pod.rates)}")
    print(f"      mecanismes  {list(cfg.pod.mechanisms)}")
    print(f"      graines     masque {list(cfg.pod.mask_seeds)}")

    found = locate(cfg, kind=MAIN)
    if not found:
        raise SystemExit(
            f"aucun run '{reg}_{MAIN}_*' dans '{exp}'.\n"
            f"  -> uv run runner.py --config-name={reg} dataset.name={market}"
        )
    print(f"      {len(found)} run(s) principal(aux) : "
          f"{[u for *_, u in found]}")

    d = load_market(market, cfg.dataset.processed_dir)
    out = Path(cfg.pod.out_dir) / exp
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    rows = []
    for client, run, uid in found:
        sweep_one(cfg, d, device, client, run, uid, rows, ckpt_dir)

    if cfg.pod.run_ablation:
        sweep_ablated(cfg, d, device, rows, ckpt_dir)

    # -- sorties ------------------------------------------------------------
    tag = f"{reg}_{market}"
    df = pd.DataFrame(rows)
    df.to_csv(out / f"raw_{tag}.csv", index=False)

    agg = (df[df.variant == "full"]
           .groupby(["market", "model", "mechanism", "rate"])
           .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                rMAE_mean=("rMAE", "mean"), rMAE_std=("rMAE", "std"),
                MAE_spike_mean=("MAE_spike", "mean"),
                MAE_spike_std=("MAE_spike", "std"),
                rate_eff=("masking_rate_effective", "mean"),
                n=("MAE", "size"))
           .reset_index())
    agg.to_csv(out / f"aggregated_{tag}.csv", index=False)

    with open(out / f"meta_{tag}.json", "w") as f:
        json.dump({"experiment": exp, "uids": [u for *_, u in found],
                   "config": OmegaConf.to_container(cfg, resolve=True)},
                  f, indent=2)

    print(f"\n[POD] {len(rows)} evaluations -> {out}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()