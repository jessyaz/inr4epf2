

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from omegaconf import OmegaConf


class BaseForecaster(nn.Module, ABC):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lookback = cfg.window.lookback
        self.horizon = cfg.window.horizon

    # -- a implementer ------------------------------------------------------

    @abstractmethod
    def forward_step(self, batch, device):
        """Retourne la prevision, de forme (B, horizon).

        Les modeles qui consomment l'observation partielle lisent `mask` ;
        ceux qui exigent une entree complete imputent d'abord P_look.
        """

    @abstractmethod
    def configure_optimizer(self):
        """Optimiseur, ou None pour une estimation en forme close."""

    # -- optionnel ----------------------------------------------------------

    def set_epoch(self, epoch):
        """Appele en debut de chaque epoch. A surcharger si besoin."""

    def fit(self, loaders, device, logger=None):
        """Estimation en forme close. A surcharger par les modeles sans gradient."""
        raise NotImplementedError(
            f"{type(self).__name__}.configure_optimizer() retourne None "
            f"mais fit() n'est pas implemente"
        )

    # -- persistance --------------------------------------------------------

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "model_cfg": OmegaConf.to_container(self.cfg.model, resolve=True),
        }, path)

    @classmethod
    def load(cls, cfg, path, map_location="cpu"):
        from omegaconf import OmegaConf
        d = torch.load(path, map_location=map_location, weights_only=False)

        if isinstance(d, dict) and "model_cfg" in d:
            saved = d["model_cfg"]
            current = OmegaConf.to_container(cfg.model, resolve=True)
            # certains reglages ne changent aucun parametre (revin,
            # pe_learnable...) : le state_dict se charge sans erreur alors
            # que le modele ne fait pas la meme chose
            diff = {k for k in set(saved) | set(current)
                    if saved.get(k) != current.get(k)}
            if diff:
                raise ValueError(
                    f"config du checkpoint differente sur {sorted(diff)} ; "
                    f"le modele reconstruit ne correspond pas aux poids"
                )
            sd = d["state_dict"]
        else:
            sd = d

        m = cls(cfg)
        m.load_state_dict(sd)
        return m

    # -- utilitaires --------------------------------------------------------

    @staticmethod
    def unpack(batch, device):
        """Deplace le batch et retourne les tenseurs dans l'ordre usuel."""
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return b["P_look"], b["mask"], b["X_look"], b["X_fut"], b["Y"]

    def check_output(self, pred, batch):
        expected = (batch["Y"].shape[0], self.horizon)
        if tuple(pred.shape) != expected:
            raise ValueError(
                f"{type(self).__name__}.forward_step doit retourner "
                f"{expected}, obtenu {tuple(pred.shape)}"
            )
        return pred