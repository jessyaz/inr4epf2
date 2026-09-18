"""Resume des balayages POD : ce qu'il faut regarder en premier.

Lit les raw_*.csv produits par le balayage et repond a quatre questions :
  - ou part chaque modele, et ou arrive-t-il ?
  - a quel taux les courbes se croisent-elles ?
  - quelle est la pente de degradation de chacun ?
  - l'avantage est-il plus marque sur les extremes que sur l'erreur globale ?

Usage :
    uv run python summarize_pod.py
    uv run python summarize_pod.py --market DE --mechanism mcar
    uv run python summarize_pod.py --reference lear --metric MAE
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

MODEL_ORDER = ["naive", "lear", "dnn", "epf_transformer", "inr"]
MARKET_ORDER = ["NP", "PJM", "BE", "FR", "DE"]


def _order(values, ref):
    known = [v for v in ref if v in values]
    return known + sorted(set(values) - set(known))


def load(results_dir):
    files = glob.glob(os.path.join(results_dir, "*", "raw_*.csv"))
    if not files:
        raise SystemExit(f"aucun raw_*.csv sous {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "variant" in df:
        df = df[df.variant == "full"]
    return df


def agg(df, metric):
    return (df.groupby(["market", "model", "mechanism", "rate"])[metric]
            .agg(mean="mean", std="std", n="size").reset_index())


def series(a, market, model, mechanism):
    """Courbe d'un bras ; le point a rate=0 porte le mecanisme 'none' et
    doit apparaitre sur toutes les courbes."""
    s = a[(a.market == market) & (a.model == model)
          & (a.mechanism.isin([mechanism, "none"]))].sort_values("rate")
    return s.rate.values, s["mean"].values, s["std"].values


# ---------------------------------------------------------------------------

def table_levels(a, mechanism, rates):
    """Niveau de chaque modele a quelques taux."""
    rows = []
    for mk in _order(a.market.unique(), MARKET_ORDER):
        for md in _order(a.model.unique(), MODEL_ORDER):
            r, v, s = series(a, mk, md, mechanism)
            if len(r) == 0:
                continue
            row = {"market": mk, "model": md}
            for t in rates:
                i = np.where(r == t)[0]
                row[f"{t:g}"] = f"{v[i[0]]:.2f}" if len(i) else "--"
            rows.append(row)
    return pd.DataFrame(rows)


def table_slope(a, mechanism, upto=0.9):
    """Degradation relative de rate=0 a `upto`. C'est le chiffre qui separe
    les profils : plat pour une ingestion masquee, croissant pour une
    imputation."""
    rows = []
    for mk in _order(a.market.unique(), MARKET_ORDER):
        for md in _order(a.model.unique(), MODEL_ORDER):
            r, v, _ = series(a, mk, md, mechanism)
            if 0.0 not in r or upto not in r:
                continue
            a0 = v[list(r).index(0.0)]
            a1 = v[list(r).index(upto)]
            asym = v[list(r).index(1.0)] if 1.0 in r else np.nan
            rows.append({"market": mk, "model": md,
                         "r=0": round(a0, 2), f"r={upto:g}": round(a1, 2),
                         "degr_%": round(100 * (a1 - a0) / a0, 1),
                         "asympt": round(asym, 2) if asym == asym else None})
    return pd.DataFrame(rows)


def table_retention(a, mechanism, rates=(0.5, 0.7, 0.9, 0.95)):
    """Fraction du chemin parcouru vers l'asymptote.

        rho(p) = [MAE(p) - MAE(0)] / [MAE(1) - MAE(0)]

    Vaut 0 sans masquage et 1 lorsque plus rien n'est observe. Un modele a
    rho = 0.6 pour p = 0.9 n'exploite plus que 40 % de ce que le lookback
    pouvait encore lui apporter ; un modele a rho = 0.1 en exploite encore
    90 %.

    L'interet de cette quantite est d'etre sans unite et interne a chaque
    modele : elle compare l'exploitation de l'information residuelle sans
    dependre du niveau d'erreur, donc reste lisible sur les marches ou les
    modeles ne partent pas du meme point.
    """
    rows = []
    for mk in _order(a.market.unique(), MARKET_ORDER):
        for md in _order(a.model.unique(), MODEL_ORDER):
            r, v, _ = series(a, mk, md, mechanism)
            if 0.0 not in r or 1.0 not in r:
                continue
            a0 = v[list(r).index(0.0)]
            a1 = v[list(r).index(1.0)]
            span = a1 - a0
            # asymptote trop proche du point nominal : le ratio n'a plus
            # de sens, le lookback n'apportait deja presque rien
            if span <= 0.1 * a0:
                continue
            row = {"market": mk, "model": md}
            for p in rates:
                i = np.where(r == p)[0]
                row[f"rho({p:g})"] = (round(float((v[i[0]] - a0) / span), 3)
                                      if len(i) else None)
            rows.append(row)
    return pd.DataFrame(rows)


def table_crossing(a, mechanism, reference):
    """Taux a partir duquel chaque modele passe sous la reference.

    Interpolation lineaire entre les deux taux encadrants ; 0 si le modele
    est devant des le depart, NaN s'il ne passe jamais devant.
    """
    rows = []
    for mk in _order(a.market.unique(), MARKET_ORDER):
        r0, v0, _ = series(a, mk, reference, mechanism)
        if len(r0) == 0:
            continue
        for md in _order(a.model.unique(), MODEL_ORDER):
            if md == reference:
                continue
            r, v, _ = series(a, mk, md, mechanism)
            common = np.intersect1d(r, r0)
            if len(common) < 2:
                continue
            d = np.interp(common, r, v) - np.interp(common, r0, v0)

            x = 0.0 if d[0] <= 0 else np.nan
            if d[0] > 0:
                for i in range(1, len(common)):
                    if d[i - 1] > 0 >= d[i]:
                        x = common[i - 1] + (common[i] - common[i - 1]) \
                            * d[i - 1] / (d[i - 1] - d[i])
                        break
            rows.append({"market": mk, "model": md,
                         f"crossing_vs_{reference}":
                             "--" if x != x else f"{x:.2f}"})
    return pd.DataFrame(rows)


def table_spike(df, mechanism, rate, metric="MAE"):
    """Avantage relatif sur l'erreur globale vs sur le decile superieur.

    L'imputation lisse : elle attenue les extremes en premier. Si l'ecart
    est plus grand sur MAE_spike que sur MAE, c'est la signature de cet
    effet."""
    spike = f"{metric}_spike"
    if spike not in df.columns:
        return pd.DataFrame()
    a1, a2 = agg(df, metric), agg(df, spike)
    rows = []
    for mk in _order(a1.market.unique(), MARKET_ORDER):
        sel1 = a1[(a1.market == mk) & (a1.mechanism == mechanism)
                  & (a1.rate == rate)]
        sel2 = a2[(a2.market == mk) & (a2.mechanism == mechanism)
                  & (a2.rate == rate)]
        for md in _order(sel1.model.unique(), MODEL_ORDER):
            g = sel1[sel1.model == md]
            s = sel2[sel2.model == md]
            if g.empty or s.empty:
                continue
            rows.append({"market": mk, "model": md,
                         metric: round(float(g["mean"].iloc[0]), 2),
                         spike: round(float(s["mean"].iloc[0]), 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/pod")
    ap.add_argument("--metric", default="MAE")
    ap.add_argument("--mechanism", default="mcar", choices=["mcar", "block"])
    ap.add_argument("--reference", default="lear")
    ap.add_argument("--market", default=None,
                    help="restreindre a un marche")
    ap.add_argument("--rates", nargs="+", type=float,
                    default=[0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0])
    ap.add_argument("--spike-rate", type=float, default=0.7)
    args = ap.parse_args()

    df = load(args.results)
    if args.market:
        df = df[df.market == args.market]

    a = agg(df, args.metric)
    pd.set_option("display.width", 200)

    n_by = df.groupby(["market", "model"]).size()
    print(f"{len(df)} evaluations | marches {sorted(df.market.unique())} "
          f"| modeles {sorted(df.model.unique())}")
    print(f"mecanisme affiche : {args.mechanism}\n")

    print(f"=== {args.metric} par taux ===")
    print(table_levels(a, args.mechanism, args.rates).to_string(index=False))

    print(f"\n=== degradation 0 -> 90 % ===")
    print(table_slope(a, args.mechanism).to_string(index=False))

    print(f"\n=== information residuelle exploitee "
          f"(rho = chemin parcouru vers l'asymptote) ===")
    ret = table_retention(a, args.mechanism)
    if not ret.empty:
        print(ret.to_string(index=False))
        print("  rho proche de 0 : le modele tire encore parti de ce qui "
              "reste observe")
        print("  rho proche de 1 : il a deja atteint sa performance sans "
              "historique")
    else:
        print("  (necessite les points a rate = 0 et rate = 1)")

    if args.reference in set(a.model):
        print(f"\n=== taux de croisement avec {args.reference} ===")
        cr = table_crossing(a, args.mechanism, args.reference)
        if not cr.empty:
            print(cr.to_string(index=False))

    sp = table_spike(df, args.mechanism, args.spike_rate, args.metric)
    if not sp.empty:
        print(f"\n=== erreur globale vs decile superieur "
              f"(r={args.spike_rate:g}) ===")
        print(sp.to_string(index=False))

    # dispersion : un effet plus petit que l'ecart-type n'est pas mesurable
    print("\n=== dispersion inter-graines (ecart-type moyen) ===")
    d = (a[a.rate > 0].groupby(["market", "model"])["std"].mean()
         .round(3).reset_index().rename(columns={"std": "std_moyen"}))
    d["n_par_point"] = [int(a[(a.market == r.market) & (a.model == r.model)]
                            ["n"].max()) for r in d.itertuples()]
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()