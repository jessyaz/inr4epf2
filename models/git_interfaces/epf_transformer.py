"""EPF-Transformer (Llorente Gonzalez & Portela Gonzalez, arXiv:2403.16108).

Encodeur Transformer ou chaque journee de 24 h est un jeton. Les previsions
exogenes sont projetees separement et concatenees a la sortie de l'encodeur
avant la tete de prediction.

L'architecture et le regime d'entrainement sont repris du depot des auteurs
(github.com/osllogon/epf-transformers) ; le pre-traitement et l'evaluation
sont les notres, pour que tous les bras compares voient les memes donnees.

Supervision dense decalee. Le jeton i observe les prix du jour i et recoit
les exogenes du jour i+1, pour predire les prix du jour i+1 : chaque fenetre
fournit donc autant de signaux d'apprentissage que de journees observees,
et non un seul. C'est ce qui rend un Transformer a six couches entrainable
sur les ~1400 fenetres disponibles. A l'inference seule la derniere journee
predite est retenue, ce qui correspond a l'horizon evalue -- l'evaluation
est donc identique a celle des autres bras.

Ecart assume : leurs hyperparametres par marche sont figes dans des
checkpoints et non publies dans le code. Nous reprenons les valeurs par
defaut du depot, fixees une fois et transposees a tous les marches, comme
pour les autres modeles a gradient.

Comme LEAR et le DNN, ce modele exige une entree complete : le masque est
traite par apply_strategy(), qui impute P_look avant le forward.
"""

import math

import torch
import torch.nn as nn

from models.basemodel import BaseForecaster


class PositionalEncoding(nn.Module):
    """Encodage sinusoidal fixe.

    Le depot d'origine le declare comme nn.Parameter, donc entrainable ;
    c'est vraisemblablement une inadvertance, l'encodage sinusoidal etant
    par construction non appris. Nous utilisons un buffer.
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                        # (B, S, d)
        return self.dropout(x + self.pe[: x.size(1)].unsqueeze(0))


class DailyElectricTransformer(nn.Module):
    """Un jeton par journee de 24 h ; une prediction de 24 h par jeton."""

    def __init__(self, embedding_dim=32, num_heads=8, dim_feedforward=128,
                 num_layers=6, normalize_first=False, dropout=0.2,
                 activation="relu", exog_dim=2):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.GELU()

        self.values_embeddings = nn.Sequential(
            nn.Linear(24, embedding_dim), act)

        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward, dropout,
            batch_first=True, norm_first=normalize_first,
            activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.features_embeddings = nn.Sequential(
            nn.Linear(24 * exog_dim, embedding_dim), act)

        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, dim_feedforward),
            nn.Dropout(dropout),
            act,
            nn.Linear(dim_feedforward, 24),
        )

    def forward(self, values, features):
        """values : (B, S*24) prix observes | features : (B, S*24, E)
        exogenes DECALEES d'une journee -> (B, S, 24)."""
        B = values.size(0)

        tokens = self.values_embeddings(values.reshape(B, -1, 24))
        enc = self.transformer_encoder(self.positional_encoding(tokens))

        feat = self.features_embeddings(
            features.reshape(B, enc.size(1), -1))

        return self.mlp(torch.cat([feat, enc], dim=2))


class Model(BaseForecaster):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg)
        self.name = "epf_transformer"
        c = cfg.model

        self.strategy = c.imputation
        self.use_lookback = bool(getattr(c, "use_lookback", True))

        if self.lookback % self.horizon != 0:
            raise ValueError(
                f"le lookback doit etre un multiple de l'horizon (un jeton "
                f"par journee), recu {self.lookback} / {self.horizon}"
            )

        self.net = DailyElectricTransformer(
            embedding_dim=c.embedding_dim,
            num_heads=c.num_heads,
            dim_feedforward=c.dim_feedforward,
            num_layers=c.num_layers,
            normalize_first=bool(getattr(c, "normalize_first", False)),
            dropout=getattr(c, "dropout", 0.2),
            activation=getattr(c, "activation", "relu"),
            exog_dim=c.exog_dim,
        )

    # -- mise en forme ------------------------------------------------------

    def _inputs(self, batch, device):
        """(values, features) au format du depot d'origine.

        values   : les L heures observees, soit L/24 jetons
        features : les exogenes DECALEES d'une journee, de sorte que le
                   jeton i recoive celles du jour i+1 -- le jour qu'il doit
                   predire
        """
        from datasets.loader import apply_strategy

        # ce modele n'ingere pas de trou : imputation prealable
        P, _ = apply_strategy(batch["P_look"].cpu(), batch["mask"].cpu(),
                              self.strategy)
        P = P.to(device)
        if not self.use_lookback:
            P = torch.zeros_like(P)              # modele ablate

        X = torch.cat([batch["X_look"].to(device),
                       batch["X_fut"].to(device)], dim=1)[:, self.horizon:]
        return P, X

    # -- entrainement et inference -----------------------------------------

    def loss(self, batch, device):
        """Perte sur toutes les journees de la fenetre, pas seulement
        l'horizon : c'est la supervision dense du depot d'origine."""
        values, features = self._inputs(batch, device)

        # cibles decalees d'une journee : le jour i+1 pour le jeton i. Les
        # L-24 premieres viennent du lookback, la derniere de l'horizon.
        full = torch.cat([batch["P_look"].to(device),
                          batch["Y"].to(device)], dim=1)
        targets = full[:, self.horizon:].reshape(values.size(0), -1,
                                                 self.horizon)

        return ((self.net(values, features) - targets) ** 2).mean()

    def forward_step(self, batch, device):
        values, features = self._inputs(batch, device)
        # seule la derniere journee predite correspond a l'horizon evalue
        return self.net(values, features)[:, -1]

    def configure_optimizer(self):
        c = self.cfg.model.optim
        return torch.optim.AdamW(self.parameters(), lr=c.lr,
                                 weight_decay=getattr(c, "weight_decay", 0.0))