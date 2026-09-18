"""MLP a 3 couches : modele de reference pour valider le pipeline.

Entree : prix passes (L) + masque (L) + exogenes passees et futures, aplatis.
Sortie : les `horizon` prix, predits d'un coup.

Le masque est concatene comme canal binaire. C'est donc un modele avec
remplissage constant + signalisation de l'absence, PAS une ingestion
reellement invariante a la valeur de remplissage : les positions masquees
traversent les couches. A garder en tete si on le compare a un modele
mask-aware.
"""

import torch
import torch.nn as nn

from models.basemodel import BaseForecaster


class Model(BaseForecaster):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg)
        self.name = "mlp"
        c = cfg.model

        L, H = self.lookback, self.horizon
        E = c.exog_dim                      # nb de variables exogenes (2 ici)
        hidden = c.hidden_dim
        dropout = getattr(c, "dropout", 0.0)

        self.use_mask = getattr(c, "use_mask", True)
        self.use_exog = getattr(c, "use_exog", True)
        self.fill_value = getattr(c, "fill_value", 0.0)

        in_dim = L                          # P_look
        if self.use_mask:
            in_dim += L                     # canal masque
        if self.use_exog:
            in_dim += L * E + H * E         # exogenes passees + futures

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, H),
        )

    def forward_step(self, batch, device):
        P, mask, X_look, X_fut, _ = self.unpack(batch, device)

        # les positions non observees ne doivent pas porter la valeur brute
        P = P.masked_fill(~mask, self.fill_value)

        feats = [P]
        if self.use_mask:
            feats.append(mask.to(P.dtype))
        if self.use_exog:
            feats.append(X_look.flatten(1))
            feats.append(X_fut.flatten(1))

        return self.net(torch.cat(feats, dim=-1))

    def configure_optimizer(self):
        c = self.cfg.model.optim
        return torch.optim.AdamW(self.parameters(), lr=c.lr,
                                 weight_decay=getattr(c, "weight_decay", 0.0))