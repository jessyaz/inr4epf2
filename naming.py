"""Nomenclature MLflow, centralisee.

Toute composition de nom passe par ici : aucune chaine n'est fabriquee
ailleurs dans le projet.

    experiment      icassp_{version}_{dataset}_{registry}[_{suffix}]
    run principal   {registry}_main_{uid}          -- produit par runner.py
    ablation        {registry}_ablated_{uid}       -- modele sans lookback
    sous-run POD    {registry}_{uid}_{rate}        -- un par point de grille

VERSION identifie la campagne en cours : la changer isole entierement les
nouveaux runs des precedents, sans rien effacer.

FROZEN fige la version d'un modele dont l'estimation ne depend pas de la
campagne. LEAR est deterministe et n'a pas d'hyperparametre libre -- son
lambda vient d'un critere d'information --, donc le reestimer a chaque
campagne ne produirait que du calcul perdu : ses checkpoints restent lus la
ou ils ont ete produits.

Le suffixe optionnel (cfg.exp_suffix) isole une variante dans sa propre
experience, pour que le balayage POD ne melange pas des architectures
differentes sous un meme nom.
"""

VERSION = "V4"
FROZEN = {"lear": "V3"}

MAIN = "main"
ABLATED = "ablated"


def experiment_name(cfg):
    version = FROZEN.get(cfg.registry, VERSION)
    base = f"icassp_{version}_{cfg.dataset.name}_{cfg.registry}"
    suffix = cfg.get("exp_suffix", None)
    return f"{base}_{suffix}" if suffix else base


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