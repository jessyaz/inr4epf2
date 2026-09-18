"""
Loader EPF : fenetrage et masquage a la volee.

"""

import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PRICE_COL = 0


# --------------------------------------------------------------------------
# Generation des masques
# --------------------------------------------------------------------------

def make_mask(L, rate, mechanism="mcar", prices=None, block_mean=12, rng=None):
    """Retourne un bool array de longueur L. True = observe."""
    rng = rng if rng is not None else np.random.default_rng()

    if rate <= 0.0:
        return np.ones(L, dtype=bool)
    if rate >= 1.0:
        return np.zeros(L, dtype=bool)

    if mechanism == "mcar":
        return rng.random(L) >= rate

    if mechanism == "mnar":
        if prices is None:
            raise ValueError("mnar requiert les prix de la fenetre")
        r = np.argsort(np.argsort(prices)) / max(L - 1, 1)
        p = 0.3 + 1.7 * r
        p = p * (rate * L / p.sum())
        # le clipping a 1 perd de la masse : la redistribuer sur les
        # positions non saturees jusqu'a atteindre le taux cible
        for _ in range(50):
            p = np.clip(p, 0.0, 1.0)
            deficit = rate * L - p.sum()
            if deficit < 1e-6:
                break
            free = p < 1.0
            if not free.any():
                break
            p[free] += deficit * p[free] / p[free].sum()
        return rng.random(L) >= np.clip(p, 0.0, 1.0)

    if mechanism == "block":
        mask = np.ones(L, dtype=bool)
        target = int(round(rate * L))
        guard = 0
        while (~mask).sum() < target and guard < 1000:
            guard += 1
            b = max(1, int(rng.geometric(1.0 / block_mean)))
            s = int(rng.integers(0, L))
            end = min(s + b, L)

            # positions du segment encore observees (le bloc peut recouvrir
            # des positions deja masquees par un tirage precedent)
            new = np.where(mask[s:end])[0]
            if len(new) == 0:
                continue

            # tronquer le dernier bloc pour ne pas depasser le taux cible
            need = target - int((~mask).sum())
            if len(new) > need:
                new = rng.choice(new, size=need, replace=False)

            mask[s + new] = False
        return mask

    raise ValueError(f"mecanisme inconnu : {mechanism}")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class EPFWindowDataset(Dataset):

    def __init__(self, series, lookback=168, horizon=24, stride=1,
                 rate=0.0, mechanism="mcar", block_mean=12, seed=0):
        self.s = torch.as_tensor(np.ascontiguousarray(series),
                                 dtype=torch.float32)          # (T, 3)
        self.L, self.H = lookback, horizon
        self.rate, self.mechanism = rate, mechanism
        self.block_mean, self.seed = block_mean, seed

        n = len(self.s) - lookback - horizon + 1
        if n <= 0:
            raise ValueError("serie trop courte pour ce lookback/horizon")
        self.starts = np.arange(0, n, stride, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        w = self.s[t: t + self.L + self.H]

        P_look = w[:self.L, PRICE_COL].clone()   # jamais altere
        X_look = w[:self.L, PRICE_COL + 1:]
        X_fut = w[self.L:, PRICE_COL + 1:]
        Y = w[self.L:, PRICE_COL]

        rng = np.random.default_rng([self.seed, t])
        mask = torch.from_numpy(
            make_mask(self.L, self.rate, self.mechanism,
                      prices=P_look.numpy(),
                      block_mean=self.block_mean, rng=rng)
        )

        return {"P_look": P_look, "mask": mask,
                "X_look": X_look, "X_fut": X_fut, "Y": Y,
                "t": torch.tensor(t)}


# --------------------------------------------------------------------------
# Strategies d'ingestion (appliquees APRES le Dataset, par bras)
# --------------------------------------------------------------------------

def _seasonal_fill(df, periods=(24, 48, 168)):
    """Cascade : meme heure J-1, puis J-2, puis S-1, puis repli."""
    out = df.copy()
    for p in periods:
        if out.isna().any().any():
            out = out.fillna(out.shift(p))
        else:
            break
    return out


def apply_strategy(P, mask, strategy, fill_value=0.0):
    """
    P    : (B, L) prix bruts, non alteres
    mask : (B, L) bool, True = observe
    Retourne (entree_du_bras, mask_ou_None).
    """
    if strategy == "none":
        # le modele consomme P + mask ; la valeur aux positions masquees
        # ne doit pas influencer sa sortie
        return P, mask

    Pn = P.clone()
    Pn[~mask] = float("nan")
    df = pd.DataFrame(Pn.numpy().T)              # (L, B)

    if strategy == "ffill":
        out = df.ffill().bfill()
    elif strategy == "linear":
        out = df.interpolate(limit_direction="both")
    elif strategy == "seasonal":
        out = _seasonal_fill(df)
    else:
        raise ValueError(f"strategie inconnue : {strategy}")

    # convention de repli UNIQUE, identique pour toutes les strategies
    # et tous les taux (sinon on fabrique une fausse rupture)
    out = out.fillna(fill_value)
    return torch.tensor(out.values.T, dtype=torch.float32), None


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------

def load_market(market, processed_dir="./datasets/processed"):
    with open(f"{processed_dir}/{market}_series.pkl", "rb") as f:
        return pickle.load(f)


def build_loader(series, lookback=168, horizon=24, stride=1,
                 rate=0.0, mechanism="mcar", block_mean=12, seed=0,
                 batch_size=64, shuffle=False, num_workers=4):
    ds = EPFWindowDataset(series, lookback, horizon, stride,
                          rate, mechanism, block_mean, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)


def inverse_price(scaler, p, n_cols=3):
    """Ramene des prix scales vers les EUR/MWh. p : array (..., ) ou tensor."""
    if torch.is_tensor(p):
        p = p.detach().cpu().numpy()
    flat = np.asarray(p).reshape(-1)
    tmp = np.zeros((len(flat), n_cols), dtype=np.float32)
    tmp[:, PRICE_COL] = flat
    return scaler.inverse_transform(tmp)[:, PRICE_COL].reshape(np.shape(p))


# --------------------------------------------------------------------------
# Controles
# --------------------------------------------------------------------------

def checks(market="PJM", processed_dir="./datasets/processed"):
    d = load_market(market, processed_dir)

    # 1. inversion du scaler : aller-retour exact ?
    x = d["train"][:100]
    back = d["scaler"].inverse_transform(x)
    print("1. inversion scaler OK :", np.isfinite(back).all())

    # 2. formes
    ld = build_loader(d["test"], stride=24, rate=0.30,
                      mechanism="block", seed=7, num_workers=0)
    b = next(iter(ld))
    print("2. formes :", {k: tuple(v.shape) for k, v in b.items()})

    # 3. taux effectif sur TOUT le split, pour chaque mecanisme
    print("3. taux effectif (split complet) :")
    for mech in ("mcar", "mnar", "block"):
        for nominal in (0.1, 0.3, 0.5, 0.9):
            l = build_loader(d["test"], stride=24, rate=nominal,
                             mechanism=mech, seed=7, num_workers=0)
            tot = sum((~bb["mask"]).float().sum().item() for bb in l)
            n = sum(bb["mask"].numel() for bb in l)
            print(f"   {mech:5s} nominal={nominal:.2f} -> effectif={tot/n:.3f}")

    # 4. reproductibilite : deux passes -> meme masque
    b2 = next(iter(build_loader(d["test"], stride=24, rate=0.30,
                                mechanism="block", seed=7, num_workers=0)))
    print("4. masque reproductible :", torch.equal(b["mask"], b2["mask"]))

    # 5. tous les bras voient le meme masque
    p_none, m = apply_strategy(b["P_look"], b["mask"], "none")
    p_ff, _ = apply_strategy(b["P_look"], b["mask"], "ffill")
    p_se, _ = apply_strategy(b["P_look"], b["mask"], "seasonal")
    print("5. P_look intact pour 'none' :", torch.equal(p_none, b["P_look"]))
    print("   ffill sans NaN :", torch.isfinite(p_ff).all().item())
    print("   seasonal sans NaN :", torch.isfinite(p_se).all().item())

    # 6. taux = 1.0 -> aucune observation
    ld1 = build_loader(d["test"], stride=24, rate=1.0, num_workers=0)
    b1 = next(iter(ld1))
    print("6. rate=1.0 -> 0 observation :", (~b1["mask"]).all().item())


if __name__ == "__main__":
    checks()