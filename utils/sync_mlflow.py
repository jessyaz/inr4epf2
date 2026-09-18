"""Rejoue les runs enregistres hors ligne vers le serveur MLflow.

Usage :
    python sync_mlflow.py                 # synchronise tout ce qui est en attente
    python sync_mlflow.py --dry-run       # liste sans envoyer
    python sync_mlflow.py --keep          # ne retire pas le tag apres envoi

Les runs deja synchronises portent le tag pending_sync=false et sont ignores.
"""

import argparse
import os
import tempfile

import mlflow
from dotenv import load_dotenv
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

OFFLINE_URI = f"sqlite:///{os.path.abspath('./mlruns_offline/mlflow.db')}"
BATCH = 900          # limite cote serveur : 1000 entites par log_batch


def pending_runs(local):
    """Tous les runs locaux encore marques a synchroniser."""
    out = []
    for exp in local.search_experiments():
        runs = local.search_runs(
            [exp.experiment_id],
            filter_string="tags.pending_sync = 'true'",
            max_results=5000,
        )
        out.extend((exp.name, r) for r in runs)
    return out


def copy_run(local, remote, exp_name, run):
    """Recree un run distant a l'identique : params, tags, metriques, artefacts."""
    exp = remote.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp else remote.create_experiment(exp_name)

    rid = run.info.run_id
    new = remote.create_run(
        experiment_id=exp_id,
        start_time=run.info.start_time,
        run_name=run.info.run_name,
    )
    nid = new.info.run_id

    params = [Param(k, str(v)) for k, v in run.data.params.items()]
    tags = [RunTag(k, str(v)) for k, v in run.data.tags.items()
            if not k.startswith("mlflow.") and k != "pending_sync"]
    tags.append(RunTag("synced_from", rid))

    # historique complet des metriques, pas seulement la derniere valeur
    metrics = []
    for key in run.data.metrics:
        for m in local.get_metric_history(rid, key):
            metrics.append(Metric(m.key, m.value, m.timestamp, m.step))

    remote.log_batch(nid, metrics=[], params=params, tags=tags)
    for i in range(0, len(metrics), BATCH):
        remote.log_batch(nid, metrics=metrics[i:i + BATCH])

    # artefacts
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for a in local.list_artifacts(rid):
                p = local.download_artifacts(rid, a.path, tmp)
                if os.path.isdir(p):
                    remote.log_artifacts(nid, p, artifact_path=a.path)
                else:
                    remote.log_artifact(nid, p)
        except Exception as e:
            print(f"    artefacts non copies : {e}")

    remote.set_terminated(nid, status=run.info.status,
                          end_time=run.info.end_time)
    return nid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="ne pas marquer les runs locaux comme synchronises")
    args = ap.parse_args()

    load_dotenv("./.env", override=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("MLFLOW_TRACKING_URI non defini")

    local = MlflowClient(tracking_uri=OFFLINE_URI)
    remote = MlflowClient(tracking_uri=uri)

    todo = pending_runs(local)
    if not todo:
        print("rien a synchroniser")
        return

    print(f"{len(todo)} run(s) en attente -> {uri}")
    for exp_name, run in todo:
        label = run.info.run_name or run.info.run_id[:8]
        if args.dry_run:
            print(f"  [dry] {exp_name} / {label}")
            continue
        try:
            nid = copy_run(local, remote, exp_name, run)
            if not args.keep:
                local.set_tag(run.info.run_id, "pending_sync", "false")
            print(f"  OK  {exp_name} / {label} -> {nid[:8]}")
        except Exception as e:
            print(f"  ECHEC {exp_name} / {label} : {e}")


if __name__ == "__main__":
    main()