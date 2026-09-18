"""Nomenclature MLflow, centralisee.

Toute composition de nom passe par ici : aucune chaine n'est fabriquee
ailleurs dans le projet.

    experiment      icassp_{dataset}_{registry}
    run principal   {registry}_main_{uid}          -- produit par runner.py
    ablation        {registry}_ablated_{uid}       -- modele sans lookback
    sous-run POD    {registry}_{uid}_{rate}        -- un par point de grille
"""

MAIN = "main"
ABLATED = "ablated"


def experiment_name(cfg):
    return f"icassp_V1_{cfg.dataset.name}_{cfg.registry}"


def main_run(registry, uid):
    return f"{registry}_{MAIN}_{uid}"


def ablated_run(registry, uid):
    return f"{registry}_{ABLATED}_{uid}"


def child_run(registry, uid, rate):
    return f"{registry}_{uid}_{rate}"


def parse_uid(run_name, registry):
    """Extrait l'uid d'un nom de run principal ou ablate, sinon None."""
    for kind in (MAIN, ABLATED):
        prefix = f"{registry}_{kind}_"
        if run_name and run_name.startswith(prefix):
            return run_name[len(prefix):]
    return None