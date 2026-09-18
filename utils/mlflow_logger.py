"""Logger MLflow tolerant aux pannes, et acces aux runs enregistres.

Comportement du logger, entierement gere ici :
  - mlflow.enabled = false       -> tous les appels sont des no-op silencieux
  - serveur injoignable au start -> ecriture directe dans le store local
  - coupure EN COURS de run      -> bascule locale et rejeu de l'historique
                                    deja emis, de facon transparente

Le runner appelle toujours les memes methodes, sans condition.
Les runs ecrits localement portent le tag pending_sync=true ; sync_mlflow.py
les rejoue vers le serveur plus tard.

Le module expose aussi de quoi retrouver et telecharger un run estime :
find_runs / fetch_checkpoint / download_checkpoint. Ces fonctions sont
appelees hors de tout MLflowLogger, donc elles chargent le .env
elles-memes -- sans quoi MLFLOW_TRACKING_URI n'est pas encore defini et
seul le store local serait interroge.
"""

import os
import socket
from urllib.parse import urlparse

# Doit etre defini AVANT l'import de mlflow : sinon les valeurs par defaut
# (timeout 120 s, 7 retries) bloquent le run plusieurs minutes quand le
# serveur tombe, et _safe n'est jamais atteint.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "0")

import mlflow
from dotenv import load_dotenv
from omegaconf import OmegaConf

# le file store ('file:./mlruns') est en mode maintenance et refuse les
# ecritures ; SQLite est le backend local recommande
OFFLINE_DB = os.path.abspath("./mlruns_offline/mlflow.db")
OFFLINE_ARTIFACTS = os.path.abspath("./mlruns_offline/artifacts")
OFFLINE_URI = f"sqlite:///{OFFLINE_DB}"


def _server_reachable(uri, timeout=3.0):
    if not uri or uri.startswith(("file:", "sqlite:")):
        return False
    p = urlparse(uri)
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        with socket.create_connection((p.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


# ===========================================================================
# Logger
# ===========================================================================

class MLflowLogger:
    """Toute methode publique est sure : elle n'interrompt jamais le run."""

    def __init__(self, cfg):
        load_dotenv("./.env", override=True)
        self.cfg = cfg
        self.mlcfg = cfg.mlflow

        self.enabled = bool(getattr(self.mlcfg, "enabled", True))
        self.force_offline = bool(getattr(self.mlcfg, "force_offline", False))
        self.experiment_name = self.mlcfg.experiment_name
        self.run_name = getattr(self.mlcfg, "run_name", None) or cfg.model_uid

        self.offline = False
        self.dead = False              # plus aucun backend disponible
        self.tester_flag = False

        # historique rejoue en cas de bascule en cours de run
        self._metrics = []             # (dict, step)
        self._artifacts = []           # chemins locaux
        self._params = {}

    # -- cycle de vie -------------------------------------------------------

    def __enter__(self):
        if not self.enabled:
            print("[mlflow] desactive")
            return self

        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not self.force_offline and _server_reachable(uri):
            if self._start(uri):
                print(f"[mlflow] serveur : {uri}")
                return self
            print("[mlflow] demarrage distant impossible")
        else:
            reason = ("force_offline" if self.force_offline
                      else f"injoignable ({uri or 'URI non definie'})")
            print(f"[mlflow] {reason}")

        self._go_offline(replay=False)
        return self

    def __exit__(self, exc_type, *args):
        if not self.enabled or self.dead:
            return False
        status = "FINISHED" if (self.tester_flag and exc_type is None) else "FAILED"
        try:
            mlflow.end_run(status)
        except Exception:
            pass
        if self.offline:
            print("[mlflow] run ecrit localement, a synchroniser "
                  "(python sync_mlflow.py)")
        return False

    # -- backend ------------------------------------------------------------

    def _start(self, uri):
        """Ouvre un run sur `uri`. Retourne False si le backend refuse."""
        try:
            mlflow.set_tracking_uri(uri)

            if uri == OFFLINE_URI:
                os.makedirs(os.path.dirname(OFFLINE_DB), exist_ok=True)
                os.makedirs(OFFLINE_ARTIFACTS, exist_ok=True)
                # le store local a son propre repertoire d'artefacts ; sans
                # cela il heriterait d'une location S3 inaccessible hors ligne
                if mlflow.get_experiment_by_name(self.experiment_name) is None:
                    mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=f"file://{OFFLINE_ARTIFACTS}",
                    )

            mlflow.set_experiment(self.experiment_name)
            try:
                mlflow.end_run()
            except Exception:
                pass
            mlflow.start_run(run_name=self.run_name)

            flat = OmegaConf.to_container(self.cfg, resolve=True)
            self._params = {
                "model_uid": flat.get("model_uid"),
                "registry": flat.get("registry"),
                "market": flat.get("dataset", {}).get("name"),
                "seed": flat.get("seed"),
                "masking_rate": flat.get("masking", {}).get("rate"),
                "masking_mechanism": flat.get("masking", {}).get("mechanism"),
            }
            mlflow.log_params(self._params)
            mlflow.log_dict(flat, "config.yaml")
            return True
        except Exception as e:
            print(f"[mlflow] erreur au demarrage : {e}")
            return False

    def _go_offline(self, replay=True):
        """Bascule sur le store local et rejoue ce qui a deja ete emis."""
        if self.offline:
            return
        self.offline = True

        # on abandonne le run distant sans tenter de le cloturer : le serveur
        # est injoignable, chaque appel couterait un nouveau timeout
        try:
            if mlflow.active_run():
                mlflow.end_run("FAILED")
        except Exception:
            pass
        finally:
            try:
                mlflow.tracking.fluent._active_run_stack.clear()
            except Exception:
                pass

        if not self._start(OFFLINE_URI):
            self.dead = True
            print("[mlflow] store local inaccessible, journalisation abandonnee")
            return

        try:
            mlflow.set_tag("pending_sync", "true")
            if replay:
                for d, step in self._metrics:
                    mlflow.log_metrics(d, step=step)
                for p in self._artifacts:
                    if os.path.exists(p):
                        mlflow.log_artifact(p)
                print(f"[mlflow] bascule locale, {len(self._metrics)} "
                      f"point(s) rejoue(s) dans {OFFLINE_URI}")
            else:
                print(f"[mlflow] ecriture locale dans {OFFLINE_URI}")
        except Exception as e:
            self.dead = True
            print(f"[mlflow] rejeu impossible : {e}")

    def _safe(self, fn):
        """Execute fn ; sur echec reseau, bascule locale puis reessaie une fois."""
        if not self.enabled or self.dead:
            return
        try:
            fn()
        except Exception as e:
            if self.offline:
                self.dead = True
                print(f"[mlflow] ecriture locale impossible : {e}")
                return
            print(f"[mlflow] perte du serveur ({e})")
            self._go_offline(replay=True)
            if not self.dead:
                try:
                    fn()
                except Exception as e2:
                    self.dead = True
                    print(f"[mlflow] echec apres bascule : {e2}")

    # -- API ----------------------------------------------------------------

    def log_metrics(self, loss_dict, epoch, prefix="train"):
        d = {f"{prefix}/{k}": float(v) for k, v in loss_dict.items()}
        self._metrics.append((d, epoch))
        self._safe(lambda: mlflow.log_metrics(d, step=epoch))

    def log_checkpoint(self, path):
        self._artifacts.append(path)
        self._safe(lambda: mlflow.log_artifact(path))

    def log_plot(self, fig, artifact_path="plots"):
        # une figure ne se rejoue pas : elle est fermee apres coup
        self._safe(lambda: mlflow.log_figure(fig, artifact_path))


# ===========================================================================
# Acces aux runs enregistres
# ===========================================================================

def _tracking_candidates():
    """URIs a essayer, dans l'ordre : serveur puis store local.

    Charge le .env ici : ces fonctions sont appelees hors de tout
    MLflowLogger, donc l'environnement n'est pas encore prepare.
    """
    load_dotenv("./.env", override=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    return ([uri] if _server_reachable(uri) else []) + [OFFLINE_URI]


def _run_name(run):
    """Le nom d'un run : selon la version de MLflow il vit dans info ou
    dans le tag mlflow.runName."""
    return run.info.run_name or run.data.tags.get("mlflow.runName", "") or ""


def find_runs(experiment_name, prefix, max_results=200):
    """Runs dont le nom commence par `prefix`, du plus recent au plus ancien.

    Retourne [(client, run, name), ...]. Le premier backend qui repond avec
    au moins un run gagne : un run non encore synchronise reste exploitable.
    """
    from mlflow.tracking import MlflowClient

    for uri in _tracking_candidates():
        try:
            c = MlflowClient(tracking_uri=uri)
            exp = c.get_experiment_by_name(experiment_name)
            if exp is None:
                continue
            out = [(c, r, _run_name(r))
                   for r in c.search_runs(
                    [exp.experiment_id],
                    order_by=["attributes.start_time DESC"],
                    max_results=max_results)
                   if _run_name(r).startswith(prefix)]
            if out:
                return out
        except Exception:
            continue
    return []


def download_checkpoint(client, run, dest="./ckpt", filename="model.pth"):
    """Telecharge l'artefact `filename` d'un run deja localise."""
    os.makedirs(dest, exist_ok=True)
    return client.download_artifacts(run.info.run_id, filename, dest)


def fetch_checkpoint(experiment_name, model_uid, dest="./ckpt",
                     filename="model.pth"):
    """Recupere un checkpoint par (experiment_name, model_uid)."""
    from mlflow.tracking import MlflowClient

    candidates = _tracking_candidates()
    for uri in candidates:
        try:
            c = MlflowClient(tracking_uri=uri)
            exp = c.get_experiment_by_name(experiment_name)
            if exp is None:
                continue
            runs = c.search_runs(
                [exp.experiment_id],
                filter_string=f"params.model_uid = '{model_uid}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if runs:
                return download_checkpoint(c, runs[0], dest, filename)
        except Exception:
            continue

    raise FileNotFoundError(
        f"aucun run '{experiment_name}' / model_uid='{model_uid}' "
        f"(cherche dans : {candidates})"
    )