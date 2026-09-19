"""Transformer a attention masquee : baseline a ingestion directe.

Un jeton par heure observee. Les positions non observees sont exclues du
softmax d'attention (`src_key_padding_mask`), donc aucune tete ne les
regarde et leur contenu n'influence pas la sortie : c'est le meme principe
que les sequences de longueur variable en traitement du langage. Aucune
imputation n'est necessaire.

Cette baseline existe pour repondre a une question que les modeles a
imputation ne permettent pas de poser : le comportement observe sous
observabilite partielle tient-il a l'ingestion directe en soi, ou a la
maniere particuliere dont notre modele la realise ? Elle consomme la meme
information que les autres bras et differe du modele propose par sa seule
architecture.

Ecart assume avec l'EPF-Transformer (Llorente & Portela), dont un jeton
represente une journee de 24 h : a cette granularite un jeton peut etre
partiellement observe, et le masque horaire n'a plus de traduction
naturelle. Le jeton horaire rend l'exclusion exacte, au prix d'une sequence
de 168 pas au lieu de 7.

Les previsions exogenes du jour predit sont projetees separement et
concatenees a la representation agregee avant la tete, comme dans
l'EPF-Transformer.
"""

import math

import torch
import torch.nn as nn

from models.basemodel import BaseForecaster


class SinusoidalPE(nn.Module):
    """Encodage positionnel fixe, non entrainable."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                       # (B, S, d)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class Model(BaseForecaster):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg)
        self.name = "masked_transformer"
        c = cfg.model

        d = c.embedding_dim
        self.use_lookback = bool(getattr(c, "use_lookback", True))

        # un jeton par heure : (prix, temps normalise)
        self.value_embed = nn.Linear(2, d)
        self.pos_enc = SinusoidalPE(d, max_len=self.lookback + 1)

        layer = nn.TransformerEncoderLayer(
            d, c.num_heads, c.dim_feedforward, c.dropout,
            batch_first=True, norm_first=bool(getattr(c, "normalize_first", True)),
            activation=getattr(c, "activation", "gelu"))
        self.encoder = nn.TransformerEncoder(layer, c.num_layers)

        self.exog_embed = nn.Sequential(
            nn.Linear(self.horizon * c.exog_dim, d), nn.GELU())

        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, c.dim_feedforward),
            nn.Dropout(c.dropout),
            nn.GELU(),
            nn.Linear(c.dim_feedforward, self.horizon),
        )

        self.register_buffer("_t", torch.empty(0), persistent=False)

    def _time_coords(self, device):
        if self._t.numel() == 0:
            self._t = (torch.arange(self.lookback, device=device,
                                    dtype=torch.float32)
                       / self.lookback * 2.0 - 1.0)
        return self._t

    def forward_step(self, batch, device):
        P, mask, _, X_fut, _ = self.unpack(batch, device)
        B = P.shape[0]

        t = self._time_coords(device).unsqueeze(0).expand(B, -1)
        x = self.value_embed(torch.stack([P, t], dim=-1))    # (B, L, d)
        x = self.pos_enc(x)

        if self.use_lookback:
            # True = position ignoree par l'attention. Aucune tete ne lit
            # une position non observee : la valeur qu'elle porte est sans
            # effet sur la sortie.
            pad = ~mask
            # une fenetre entierement masquee produirait un softmax sur un
            # ensemble vide, donc des NaN ; on y laisse la premiere position
            # visible, son embedding etant de toute facon ignore ensuite
            empty = pad.all(dim=1)
            if empty.any():
                pad = pad.clone()
                pad[empty, 0] = False

            enc = self.encoder(x, src_key_padding_mask=pad)

            # moyenne sur les seules positions observees
            m = mask.unsqueeze(-1).to(enc.dtype)
            n = m.sum(dim=1).clamp(min=1.0)
            z = (enc * m).sum(dim=1) / n
            z = torch.where(mask.any(dim=1, keepdim=True), z,
                            torch.zeros_like(z))
        else:
            # modele ablate : aucun prix passe
            z = torch.zeros(B, x.size(-1), device=device, dtype=x.dtype)

        e = self.exog_embed(X_fut.reshape(B, -1))
        return self.head(torch.cat([e, z], dim=-1))

    def configure_optimizer(self):
        c = self.cfg.model.optim
        return torch.optim.AdamW(self.parameters(), lr=c.lr,
                                 weight_decay=getattr(c, "weight_decay", 0.0))