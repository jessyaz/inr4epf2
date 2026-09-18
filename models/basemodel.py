

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
        """Par defaut : state_dict PyTorch. Surcharge si le modele porte
        des objets non-tensoriels (estimateurs sklearn, par exemple)."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, cfg, path, map_location="cpu"):
        m = cls(cfg)
        m.load_state_dict(torch.load(path, map_location=map_location))
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