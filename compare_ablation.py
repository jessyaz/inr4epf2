"""Compare les variantes d'ablation, identifiees par le suffixe d'experience.

Les CSV de chaque variante portent tous model = "inr" ; ce qui les distingue
est le dossier de sortie, nomme d'apres l'experience MLflow. On repart donc
du chemin pour retrouver la variante.

Deux lectures :
  - a p = 0, l'ecart mesure la contribution architecturale
  - sous masquage, un ecart qui se creuse indique que le mecanisme compte
    d'autant plus que le lookback se rarefie

Usage :
    uv run python compare_ablation.py
    uv run python compare_ablation.py --market DE --rates 0 0.5 0.9
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

LABEL = {
    None: "complet",
    "static": "exog. statique",
    "noexog": "sans exogene",
    "ablation_static": "exog. statique",
    "ablation_noexog": "sans exogene",
}
ORDER = ["complet", "exog. statique", "sans exogene"]


def variant_of(path, market, registry):
    """Le suffixe d'experience est dans le nom du dossier :
    results/pod/icassp_V1_{market}_{registry}[_{suffix}]/ ..."""
    folder = os.path.basename(os.path.dirname(path))
    m = re.match(rf".*_{market}_{registry}(?:_(.+))?$", folder)
    suffix = m.group(1) if m else None
    return LABEL.get(suffix, suffix or "complet")


def load(results_dir, market, registry):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*", f"raw_{registry}_*.csv")):
        df = pd.read_csv(f)
        df = df[df.market == market]
        if df.empty:
            continue
        if "variant" in df:
            df = df[df.variant == "full"]
        df = df.copy()
        df["arm"] = variant_of(f, market, registry)
        rows.append(df)
    if not rows:
        raise SystemExit(f"aucun CSV pour {registry} / {market} "
                         f"sous {results_dir}")
    return pd.concat(rows, ignore_index=True)


def table(df, mechanism, rates, metric="MAE"):
    a = (df.groupby(["arm", "mechanism", "rate"])[metric]
         .agg(mean="mean", std="std", n="size").reset_index())
    arms = [x for x in ORDER if x in set(a.arm)] + \
           sorted(set(a.arm) - set(ORDER))

    rows = []
    for arm in arms:
        sel = a[(a.arm == arm) & (a.mechanism.isin([mechanism, "none"]))]
        row = {"variant": arm}
        for r in rates:
            s = sel[sel.rate == r]
            if s.empty:
                row[f"{r:g}"] = "--"
                continue
            m, sd = float(s["mean"].iloc[0]), s["std"].iloc[0]
            row[f"{r:g}"] = (f"{m:.3f}" if pd.isna(sd)
                             else f"{m:.3f}±{sd:.3f}")
        rows.append(row)
    return pd.DataFrame(rows), a, arms


def deltas(a, arms, mechanism, rates, metric="MAE"):
    """Ecart de chaque variante au modele complet, en absolu et en %."""
    if "complet" not in arms:
        return pd.DataFrame()
    base = a[(a.arm == "complet") & (a.mechanism.isin([mechanism, "none"]))]
    rows = []
    for arm in arms:
        if arm == "complet":
            continue
        sel = a[(a.arm == arm) & (a.mechanism.isin([mechanism, "none"]))]
        row = {"variant": arm}
        for r in rates:
            b = base[base.rate == r]
            s = sel[sel.rate == r]
            if b.empty or s.empty:
                row[f"{r:g}"] = "--"
                continue
            d = float(s["mean"].iloc[0]) - float(b["mean"].iloc[0])
            pct = 100 * d / float(b["mean"].iloc[0])
            row[f"{r:g}"] = f"{d:+.3f} ({pct:+.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/pod")
    ap.add_argument("--market", default="DE")
    ap.add_argument("--registry", default="inr")
    ap.add_argument("--metric", default="MAE")
    ap.add_argument("--mechanism", default="mcar", choices=["mcar", "block"])
    ap.add_argument("--rates", nargs="+", type=float,
                    default=[0.0, 0.3, 0.5, 0.7, 0.9])
    args = ap.parse_args()

    df = load(args.results, args.market, args.registry)
    pd.set_option("display.width", 200)

    print(f"{args.market} | {args.metric} | {args.mechanism} | "
          f"variantes : {sorted(df.arm.unique())}\n")

    tbl, a, arms = table(df, args.mechanism, args.rates, args.metric)
    print("=== niveaux ===")
    print(tbl.to_string(index=False))

    d = deltas(a, arms, args.mechanism, args.rates, args.metric)
    if not d.empty:
        print("\n=== ecart au modele complet ===")
        print(d.to_string(index=False))
        print("\n  un ecart qui se creuse avec le taux indique que le")
        print("  mecanisme compte d'autant plus que le lookback se rarefie")

    # la dispersion decide de ce qui est concluant : un ecart plus petit
    # qu'elle ne prouve rien
    print("\n=== dispersion inter-graines ===")
    disp = (a[a.rate > 0].groupby("arm")["std"].mean().round(3)
            .reset_index().rename(columns={"std": "std_moyen"}))
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
