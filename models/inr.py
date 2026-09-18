"""INR conditionne par FiLM, a ingestion masquee.

Le lookback est encode par un DeepSets qui n'agrege que les positions
observees : aucune imputation, et la valeur portee par une position masquee
n'influence pas la sortie. L'horizon est produit par un INR sur le temps
continu, module par FiLM a partir du code de fenetre et de l'etat exogene.

Les exogenes (passees et futures) sont intactes, seul le lookback de prix
est masque.

Trois variantes sont exposees pour la campagne d'ablation. Elles sont
toutes desactivables ; la configuration par defaut correspond au modele
decrit dans le papier.

  model.revin          normalisation par fenetre sur les positions
                       OBSERVEES, avec de-normalisation de la prediction.
                       Traite le decalage de distribution a la source, mais
                       sous masquage fort mu et sigma sont estimes sur peu
                       de points : canal d'effet supplementaire a mesurer.
  model.inr.layer_norm LayerNorm apres chaque modulation FiLM. Stabilise une
                       pile profonde ou le produit des gamma peut deriver.
  model.inr.skip       connexion residuelle entre couches de l'INR.
  model.static_exog    le code exogene est fige sur tout l'horizon au lieu
                       de varier a chaque pas. C'est le conditionnement des
                       INR existants. L'information disponible est
                       IDENTIQUE -- l'etat final du LSTM a parcouru tout
                       l'horizon --, seule sa resolution temporelle change :
                       l'ablation isole donc le mecanisme, pas le contenu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseForecaster

# bande utile pour un lookback de 168 h : 84 = Nyquist
FREQ_MIN = 0.875
FREQ_MAX = 84.0


class PE(nn.Module):
    """Encodage de Fourier a frequences optionnellement apprenables."""

    def __init__(self, num_frequencies, learnable=True, logger=None):
        super().__init__()
        freqs = torch.logspace(
            torch.log2(torch.tensor(FREQ_MIN)),
            torch.log2(torch.tensor(FREQ_MAX)),
            num_frequencies,
            base=2.0,
        )
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

        self.learnable = learnable
        self.current_epoch = 0
        self.logger = logger

    def forward(self, t):
        angles = t.unsqueeze(-1) * self.freqs * 2.0
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.logger is not None and self.learnable:
            v = self.freqs.detach().cpu().numpy()
            self.logger.log_metrics({f"pe_freq_{i}": float(x)
                                     for i, x in enumerate(v)},
                                    epoch, prefix="pe")


class INR(nn.Module):
    """MLP sur le temps continu, module couche par couche par FiLM."""

    def __init__(self, cfg, logger=None):
        super().__init__()
        c = cfg.inr
        self.num_layers = c.num_layers
        self.use_layer_norm = bool(getattr(c, "layer_norm", False))
        self.use_skip = bool(getattr(c, "skip", False))

        self.fourier = PE(c.num_frequencies, learnable=c.pe_learnable,
                          logger=logger)
        dims = [2 * c.num_frequencies] + [c.hidden_dim] * c.num_layers
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                     for i in range(c.num_layers)])
        self.output_layer = nn.Linear(c.hidden_dim, c.output_dim)

        if bool(getattr(c, "siren_init", False)):
            self._siren_init()

        self.norms = nn.ModuleList(
            [nn.LayerNorm(c.hidden_dim) for _ in range(c.num_layers)]
        ) if self.use_layer_norm else None

        acts = {"silu": F.silu, "gelu": F.gelu}
        if c.activation not in acts:
            raise ValueError(f"activation inconnue : {c.activation}")
        self.activation = acts[c.activation]

    def _siren_init(self):
        """Initialisation uniforme en 1/sqrt(fan_in), facon SIREN.

        L'entree est un encodage de Fourier : ses composantes sont deja
        bornees dans [-1, 1] et fortement correlees entre elles. L'init par
        defaut de PyTorch (Kaiming, calibree pour des entrees decorrelees et
        une activation ReLU) produit alors des pre-activations de variance
        mal controlee, et certaines graines divergent en debut
        d'entrainement.
        """
        with torch.no_grad():
            for layer in self.layers:
                fan_in = layer.weight.shape[1]
                bound = (1.0 / fan_in) ** 0.5
                layer.weight.uniform_(-bound, bound)
                if layer.bias is not None:
                    layer.bias.zero_()
            fan_in = self.output_layer.weight.shape[1]
            bound = (1.0 / fan_in) ** 0.5
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()

    def set_epoch(self, epoch):
        self.fourier.set_epoch(epoch)

    def forward(self, t, film):
        """t : (N,) | film : deux tenseurs (N, num_layers, hidden_dim)."""
        gamma, beta = film
        x = self.fourier(t)

        for i, layer in enumerate(self.layers):
            h = gamma[:, i] * layer(x) + beta[:, i]
            if self.norms is not None:
                h = self.norms[i](h)
            h = self.activation(h)
            # residuel a partir de la 2e couche : la 1re change de dimension
            x = x + h if (self.use_skip and i > 0) else h

        return self.output_layer(x)


class FiLMGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers, layer_dim):
        super().__init__()
        self.num_layers, self.layer_dim = num_layers, layer_dim
        self.shared = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2 * layer_dim)
                                    for _ in range(num_layers)])
        # initialisation a l'identite : gamma = 1, beta = 0
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[:layer_dim] = 1.0

    def forward(self, z):
        h = self.shared(z)
        out = torch.stack([head(h) for head in self.heads], dim=1)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta          # (N, num_layers, layer_dim) chacun


class DeepSetsEncoder(nn.Module):
    """Agregation invariante sur les seules positions observees.

    La sortie est normalisee (LayerNorm) : le niveau des prix varie d'une
    fenetre a l'autre, et sans cela l'echelle du code de fenetre derive avec
    lui, ce qui deplace le point de fonctionnement du FiLM.
    """

    def __init__(self, input_dim, hidden_dim, norm_output=True):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(2 * hidden_dim) if norm_output else None

    def forward(self, elements, mask):
        e = self.phi(elements)
        m = mask.unsqueeze(-1).to(e.dtype)

        # les positions masquees ne contribuent ni a la moyenne ni au max :
        # la valeur qu'elles portent n'a aucun effet sur la sortie
        mean_pool = (e * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        max_pool = e.masked_fill(m == 0, float("-inf")).max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)   # fenetre vide

        z = torch.cat([mean_pool, max_pool], dim=-1)
        return self.norm(z) if self.norm is not None else z


def masked_stats(P, mask, eps=1e-3):
    """Moyenne et ecart-type sur les seules positions observees. (B, 1)"""
    m = mask.to(P.dtype)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu = (P * m).sum(dim=1, keepdim=True) / n
    var = (((P - mu) * m) ** 2).sum(dim=1, keepdim=True) / n
    return mu, var.sqrt().clamp(min=eps)


class Model(BaseForecaster):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg)
        self.name = "inr"
        c = cfg.model
        self.mcfg = c

        self.use_exog = bool(getattr(c, "use_exog", True))
        self.use_lookback = bool(getattr(c, "use_lookback", True))
        self.static_exog = bool(getattr(c, "static_exog", False))
        self.revin = bool(getattr(c, "revin", False))
        self.norm_deepsets = bool(getattr(c, "norm_deepsets", True))

        self.deepsets_encoder = DeepSetsEncoder(
            c.deepsets.input_dim, c.deepsets.hidden_dim,
            norm_output=self.norm_deepsets,
        )

        z_dim = 2 * c.deepsets.hidden_dim if self.use_lookback else 0
        if self.use_exog:
            self.lstm_past = nn.LSTM(c.lstm.input_dim, c.lstm.hidden_dim,
                                     batch_first=True)
            self.lstm_future = nn.LSTM(c.lstm.input_dim, c.lstm.hidden_dim,
                                       batch_first=True)
            z_dim += c.lstm.hidden_dim
        else:
            self.lstm_past = self.lstm_future = None

        if z_dim == 0:
            raise ValueError("use_exog et use_lookback tous deux desactives")

        self.inr = INR(c, logger)
        self.film_generator = FiLMGenerator(
            z_dim=z_dim,
            hidden_dim=c.inr.film_hidden_dim,
            num_layers=c.inr.num_layers,
            layer_dim=c.inr.hidden_dim,
        )

        # echelle temporelle, fixe : mise en cache au premier appel
        self.register_buffer("_t_past", torch.empty(0), persistent=False)
        self.register_buffer("_t_future", torch.empty(0), persistent=False)

    def set_epoch(self, epoch):
        self.inr.set_epoch(epoch)

    def _time_scale(self, device):
        if self._t_past.numel() == 0:
            idx = torch.arange(-self.lookback, self.horizon,
                               device=device, dtype=torch.float32)
            t = idx / self.lookback * torch.pi
            self._t_past = t[:self.lookback]
            self._t_future = t[self.lookback:]
        return self._t_past, self._t_future

    def forward_step(self, batch, device):
        P, mask, X_look, X_fut, _ = self.unpack(batch, device)
        B, H = P.shape[0], self.horizon

        t_past, t_future = self._time_scale(device)

        mu = sd = None
        if self.revin and self.use_lookback:
            # statistiques calculees sur les seules positions observees
            mu, sd = masked_stats(P, mask)
            P = (P - mu) / sd

        parts = []

        if self.use_lookback:
            # (B, L, 2) : chaque element est un couple (temps, prix observe)
            elems = torch.stack([t_past.unsqueeze(0).expand(B, -1), P], dim=-1)
            parts.append(self.deepsets_encoder(elems, mask))   # (B, 2*d)

        if self.use_exog:
            _, (h, c) = self.lstm_past(X_look)
            # etat exogene a chaque pas de l'horizon, en une seule passe
            h_fut, _ = self.lstm_future(X_fut, (h, c))         # (B, H, d)
            if self.static_exog:
                # ablation : un seul code pour tout l'horizon. L'etat final
                # a deja parcouru les H pas, donc le contenu informationnel
                # est le meme ; seule disparait la variation avec t.
                h_fut = h_fut[:, -1:].expand(-1, H, -1)
            parts.append(h_fut)

        # code de fenetre constant sur l'horizon, concatene a l'etat exogene
        # qui lui varie par pas
        if self.use_lookback and self.use_exog:
            z = torch.cat([parts[0].unsqueeze(1).expand(-1, H, -1),
                           parts[1]], dim=-1)
        elif self.use_exog:
            z = parts[0]
        else:
            z = parts[0].unsqueeze(1).expand(-1, H, -1)

        # les H pas sont traites en une passe plutot qu'en boucle
        films = self.film_generator(z.reshape(B * H, -1))
        pred = self.inr(t_future.repeat(B), films).reshape(B, H, -1)
        pred = pred.squeeze(-1)                                # (B, H)

        if mu is not None:
            pred = pred * sd + mu

        return pred

    def configure_optimizer(self):
        c = self.mcfg.optim
        pe, inr_layers, other = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "fourier.freqs" in name:
                pe.append(p)
            elif name.startswith(("inr.layers", "inr.output_layer")):
                inr_layers.append(p)
            else:
                other.append(p)

        groups = [{"params": other, "lr": c.lr},
                  {"params": inr_layers, "lr": c.lr_inr},
                  {"params": pe, "lr": c.lr_pe}]
        groups = [g for g in groups if g["params"]]
        return torch.optim.AdamW(groups,
                                 weight_decay=getattr(c, "weight_decay", 0.0))