"""DNN : le second modele de reference du benchmark (Lago et al. 2021).

Perceptron a deux couches cachees sur EXACTEMENT les memes regresseurs que
LEAR -- 247 entrees, 24 sorties, une passe pour les 24 heures. C'est ce qui
rend la comparaison interpretable : les deux modeles voient la meme
information, seule la forme fonctionnelle change.

    prix    J-1, J-2, J-3, J-7          4 x 24  =  96
    exog    J (previsions TSO)          E x 24  =  48   (E = 2)
    exog    J-1                         E x 24  =  48
    exog    J-7                         E x 24  =  48
    dummies jour de semaine                     =   7
                                                  ----
                                                   247

Ecart assume avec Lago : leurs hyperparametres sont optimises par TPE sur
1500 iterations et par marche. Ici ils sont fixes et identiques partout, le
meme traitement etant applique a tous les reseaux compares -- une recherche
asymetrique fausserait la comparaison plus qu'elle ne l'ameliorerait.

Comme LEAR, le DNN exige une entree complete : le masque est traite par
apply_strategy(), qui impute P_look avant la construction des features. C'est
le point du protocole -- meme famille que le modele evalue, ingestion
differente.
"""

import numpy as np
import torch
import torch.nn as nn

from models.basemodel import BaseForecaster

# decoupage d'un lookback de 168 h aligne sur les journees
DAY_SLICES = {1: slice(144, 168), 2: slice(120, 144),
              3: slice(96, 120), 7: slice(0, 24)}


class Model(BaseForecaster):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg)
        self.name = "dnn"
        c = cfg.model

        self.price_lags = list(c.price_lags)
        self.exog_lags = list(c.exog_lags)
        self.strategy = c.imputation
        self.use_dummies = bool(getattr(c, "use_dummies", True))
        self.use_lookback = bool(getattr(c, "use_lookback", True))

        if self.lookback != 168:
            raise ValueError(
                f"DNN attend un lookback de 168 h (7 jours), recu {self.lookback}"
            )

        n_in = self._n_features(c.exog_dim)
        h1, h2 = c.hidden_1, c.hidden_2
        p = getattr(c, "dropout", 0.0)

        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.GELU(), nn.Dropout(p),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(p),
            nn.Linear(h2, self.horizon),
        )
        print(f"  DNN : {n_in} -> {h1} -> {h2} -> {self.horizon}")

    def _n_features(self, exog_dim):
        n = 0
        if self.use_lookback:
            n += 24 * len(self.price_lags)
            n += 24 * exog_dim * sum(1 for l in self.exog_lags if l != 0)
        n += 24 * exog_dim * sum(1 for l in self.exog_lags if l == 0)
        if self.use_dummies:
            n += 7
        return n

    # -- features : identiques a celles de LEAR -----------------------------

    def _features(self, P, X_look, X_fut, t):
        feats = []

        # use_lookback=False : modele ablate, aucun prix passe en entree
        if self.use_lookback:
            feats += [P[:, DAY_SLICES[l]] for l in self.price_lags]

        for l in self.exog_lags:
            if l == 0:
                feats.append(X_fut.reshape(X_fut.shape[0], -1))
            elif self.use_lookback:
                feats.append(X_look[:, DAY_SLICES[l]].reshape(X_look.shape[0], -1))

        if self.use_dummies:
            # avec stride 24, deux fenetres consecutives sont deux jours
            # consecutifs : t // 24 mod 7 identifie le jour sans date
            day = (t // 24) % 7
            dow = torch.zeros(len(day), 7, dtype=P.dtype, device=P.device)
            dow[torch.arange(len(day)), day] = 1.0
            feats.append(dow)

        return torch.cat(feats, dim=-1)

    def forward_step(self, batch, device):
        from datasets.loader import apply_strategy

        # le DNN n'ingere pas de trou : imputation prealable, comme LEAR
        P, _ = apply_strategy(batch["P_look"].cpu(), batch["mask"].cpu(),
                              self.strategy)
        P = P.to(device)
        X_look = batch["X_look"].to(device)
        X_fut = batch["X_fut"].to(device)

        x = self._features(P, X_look, X_fut, batch["t"].to(device))
        return self.net(x)

    def configure_optimizer(self):
        c = self.cfg.model.optim
        return torch.optim.AdamW(self.parameters(), lr=c.lr,
                                 weight_decay=getattr(c, "weight_decay", 0.0))