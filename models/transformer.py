"""Transformer sur le lookback horaire, en deux variantes d'ingestion.

L'architecture est identique dans les deux cas ; seule change la maniere
dont les heures non observees sont traitees :

  masked_attention = true   les positions absentes sont exclues du softmax
                            d'attention. Aucune tete ne les lit, aucune
                            valeur n'est reconstruite.
  masked_attention = false  la serie est imputee en amont par
                            apply_strategy(), comme pour LEAR et le DNN,
                            puis traitee comme une sequence complete.

La paire existe pour isoler le mode d'ingestion a architecture constante :
meme nombre de parametres, memes couches, meme entrainement. L'ecart entre
les deux ne peut venir que du traitement des valeurs manquantes -- ce
qu'aucune comparaison entre familles de modeles ne permet d'etablir.

Un jeton par heure, plutot que par journee : a la granularite journaliere
un jeton peut etre partiellement observe, et le masque horaire n'a plus de
traduction exacte. Le cout est une sequence de 168 pas.

Les previsions exogenes du jour predit sont projetees separement et
concatenees a la representation agregee avant la tete.
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
        c = cfg.model

        self.masked_attention = bool(getattr(c, "masked_attention", True))
        self.strategy = getattr(c, "imputation", "linear")
        self.use_lookback = bool(getattr(c, "use_lookback", True))
        self.name = ("masked_transformer" if self.masked_attention
                     else "imputed_transformer")

        d = c.embedding_dim
        self.value_embed = nn.Linear(2, d)      # (prix, temps normalise)
        self.pos_enc = SinusoidalPE(d, max_len=self.lookback + 1)

        layer = nn.TransformerEncoderLayer(
            d, c.num_heads, c.dim_feedforward, c.dropout,
            batch_first=True,
            norm_first=bool(getattr(c, "normalize_first", True)),
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

    def _encode(self, P, mask, device):
        """Representation agregee du lookback. (B, d)"""
        B = P.shape[0]
        t = self._time_coords(device).unsqueeze(0).expand(B, -1)
        x = self.pos_enc(self.value_embed(torch.stack([P, t], dim=-1)))

        if not self.masked_attention:
            # la serie a ete imputee en amont : toutes les positions sont
            # traitees comme observees
            return self.encoder(x).mean(dim=1)

        # True = position ignoree par l'attention
        pad = ~mask
        # une fenetre entierement masquee donnerait un softmax sur un
        # ensemble vide, donc des NaN : on y laisse une position visible,
        # dont la contribution est ensuite annulee au pooling
        empty = pad.all(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False

        enc = self.encoder(x, src_key_padding_mask=pad)

        m = mask.unsqueeze(-1).to(enc.dtype)
        z = (enc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return torch.where(mask.any(dim=1, keepdim=True), z,
                           torch.zeros_like(z))

    def forward_step(self, batch, device):
        P = batch["P_look"].to(device)
        mask = batch["mask"].to(device)
        X_fut = batch["X_fut"].to(device)
        B = P.shape[0]

        if not self.masked_attention:
            from datasets.loader import apply_strategy
            P, _ = apply_strategy(batch["P_look"].cpu(),
                                  batch["mask"].cpu(), self.strategy)
            P = P.to(device)

        if self.use_lookback:
            z = self._encode(P, mask, device)
        else:
            # modele ablate : aucun prix passe
            z = torch.zeros(B, self.value_embed.out_features,
                            device=device, dtype=P.dtype)

        e = self.exog_embed(X_fut.reshape(B, -1))
        return self.head(torch.cat([e, z], dim=-1))

    def configure_optimizer(self):
        c = self.cfg.model.optim
        return torch.optim.AdamW(self.parameters(), lr=c.lr,
                                 weight_decay=getattr(c, "weight_decay", 0.0))