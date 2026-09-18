"""
Preprocessing EPF (benchmark Lago et al. 2021).
"""

import os
import pickle

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from epftoolbox.data import read_data, scaling

COLS = ["Price", "Grid load forecast", "Wind power forecast"]

# Lago et al. 2021, section 4.3.2 : 42 semaines de validation
VAL_WEEKS = 42
VAL_HOURS = VAL_WEEKS * 7 * 24


def process_market(market: str, data_dir: str, normalize_method: str):
    """Lit un marche, split train/val/test, scale sur le train seul."""
    df_train, df_test = read_data(path=data_dir, dataset=market)
    df_train.columns = COLS
    df_test.columns = COLS

    df_train_raw = df_train.iloc[:-VAL_HOURS]
    df_val_raw = df_train.iloc[-VAL_HOURS:]
    df_test_raw = df_test

    # scaling() ajuste sur le PREMIER tableau (le train) et transforme
    # les autres. Le test n'intervient jamais dans l'estimation des
    # statistiques : pas de fuite.
    (train_s, val_s, test_s), scaler = scaling(
        [df_train_raw.values, df_val_raw.values, df_test_raw.values],
        normalize=normalize_method,
    )

    return {
        "train": np.ascontiguousarray(train_s, dtype=np.float32),
        "val": np.ascontiguousarray(val_s, dtype=np.float32),
        "test": np.ascontiguousarray(test_s, dtype=np.float32),
        "dates_train": df_train_raw.index.values,
        "dates_val": df_val_raw.index.values,
        "dates_test": df_test_raw.index.values,
        "scaler": scaler,
        "columns": COLS,
        "normalize_method": normalize_method,
    }


def process_markets(data_dir: str, markets: list, normalize_method: str,
                    processed_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    out = {}
    for market in tqdm(markets, desc="Marches"):
        md = process_market(market, data_dir, normalize_method)
        out[market] = md

        path = os.path.join(processed_dir, f"{market}_series.pkl")
        with open(path, "wb") as f:
            pickle.dump(md, f, protocol=pickle.HIGHEST_PROTOCOL)

        n_tr, n_va, n_te = len(md["train"]), len(md["val"]), len(md["test"])
        tqdm.write(f"  {market}: train={n_tr}h  val={n_va}h  test={n_te}h")

    return out


def sanity_check(md: dict, lookback: int = 168, horizon: int = 24):
    """Controles a faire une fois avant de lancer quoi que ce soit."""
    for split in ("train", "val", "test"):
        a = md[split]
        assert a.ndim == 2 and a.shape[1] == 3, f"{split}: forme {a.shape}"
        assert np.isfinite(a).all(), f"{split}: NaN ou inf presents"
        assert len(a) > lookback + horizon, f"{split}: trop court"
        assert len(a) == len(md[f"dates_{split}"]), f"{split}: dates desalignees"

    # le scaler doit etre ajuste par colonne, pas globalement
    tr = md["train"]
    print("  moyennes par colonne (train) :", tr.mean(axis=0).round(3))
    print("  ecarts-types par colonne     :", tr.std(axis=0).round(3))

    # le test ne doit PAS etre centre-reduit : ses stats viennent du train
    te = md["test"]
    print("  moyennes par colonne (test)  :", te.mean(axis=0).round(3))


@hydra.main(version_base=None, config_path="./", config_name="datasets")
def main(cfg: DictConfig):
    processed_dir = "./datasets/processed"
    data = process_markets(
        data_dir=cfg.data.data_dir,
        markets=cfg.data.markets,
        normalize_method=cfg.data.normalize_method,
        processed_dir=processed_dir,
    )

    first = cfg.data.markets[0]
    print(f"\n--- Sanity check ({first}) ---")
    sanity_check(data[first])


if __name__ == "__main__":
    main()