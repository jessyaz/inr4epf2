"""EPF-Transformer (Llorente Gonzalez & Portela Gonzalez, arXiv:2403.16108).

Encodeur Transformer ou chaque journee de 24 h est un jeton. Les previsions
exogenes du jour predit sont projetees separement puis concatenees a la
sortie de l'encodeur avant la tete de prediction.

L'architecture est reprise telle quelle du depot des auteurs
(github.com/osllogon/epf-transformers, src/models.py) ; le pre-traitement,
l'entrainement et l'evaluation sont les notres, pour que tous les bras
compares voient exactement les memes donnees.

Deux ecarts a signaler :

  - leur tete produit une sortie par jeton, soit L/24 journees de 24 h. Nous
    ne gardons que le dernier jeton, celui qui suit immediatement la fenetre
    observee. Leur code suppose implicitement un lookback d'une seule
    journee ; avec 168 h il faut choisir, et le dernier jeton est le seul
    aligne sur l'horizon a predire.
  - leurs hyperparametres ne sont pas ceux publies : comme pour les autres
    modeles a gradient, ils sont fixes une fois et transposes a tous les
    marches, afin que l'effort de reglage soit le meme partout.

Comme LEAR et le DNN, ce modele exige une entree complete : le masque est
traite par apply_strategy(), qui impute P_look avant le forward.
"""

import math

import torch
import torch.nn as nn

from models.basemodel import BaseForecaster


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoidal (Vaswani et al.), tel que dans le
    depot d'origine."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div)
        pe[:, 0, 1::2] = torch.cos(position * div)
        # buffer et non Parameter : l'encodage est fixe. Le depot d'origine
        # le declare comme Parameter, ce qui le rend entrainable par
        # inadvertance.
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x : (B, S, d_model) ; pe : (max_len, 1, d_model)
        return self.dropout(x + self.pe[: x.size(1), 0].unsqueeze(0))


class DailyElectricTransformer(nn.Module):
    """Un jeton par journee de 24 h."""

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
        """values : (B, L) | features : (B, 24, E) -> (B, 24)."""
        B = values.size(0)

        # une journee = un jeton
        tokens = self.values_embeddings(values.reshape(B, -1, 24))
        enc = self.transformer_encoder(self.positional_encoding(tokens))

        feat = self.features_embeddings(features.reshape(B, 1, -1))
        feat = feat.expand(-1, enc.size(1), -1)

        out = self.mlp(torch.cat([feat, enc], dim=2))    # (B, S, 24)
        return out[:, -1]                                # dernier jeton


class Model(BaseForecaster):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg)
        self.name = "epf_transformer"
        c = cfg.model

        self.strategy = c.imputation
        self.use_lookback = bool(getattr(c, "use_lookback", True))

        if self.lookback % 24 != 0:
            raise ValueError(
                f"le lookback doit etre un multiple de 24 h (un jeton par "
                f"journee), recu {self.lookback}"
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

    def forward_step(self, batch, device):
        from datasets.loader import apply_strategy

        # ce modele n'ingere pas de trou : imputation prealable, comme LEAR
        # et le DNN
        P, _ = apply_strategy(batch["P_look"].cpu(), batch["mask"].cpu(),
                              self.strategy)
        P = P.to(device)
        X_fut = batch["X_fut"].to(device)

        if not self.use_lookback:
            # modele ablate : aucun prix passe. On conserve la structure en
            # jetons pour que l'architecture reste inchangee, mais le
            # contenu est neutre.
            P = torch.zeros_like(P)

        return self.net(P, X_fut)

    def configure_optimizer(self):
        c = self.cfg.model.optim
        return torch.optim.AdamW(self.parameters(), lr=c.lr,
                                 weight_decay=getattr(c, "weight_decay", 0.0))