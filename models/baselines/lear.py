"""LEAR : Lasso Estimated AutoRegressive (Lago et al., Applied Energy 2021).

Implementation autonome, sans dependance a epftoolbox.models.LEAR.

Pourquoi ne pas utiliser le toolbox : son LEAR appelle LassoLarsIC en
comptant sur normalize=True, defaut de scikit-learn <= 1.1. Le parametre est
passe a False en 1.2 puis supprime en 1.4, si bien que sur sklearn moderne
les regresseurs ne sont plus normalises et la selection de lambda devient
inadaptee. C'est la cause des tickets ouverts #10 et #15 du depot. Ici la
normalisation est donc faite explicitement.

Structure des regresseurs, pour predire le jour J (fenetres alignees, stride 24) :
    prix    J-1, J-2, J-3, J-7          4 x 24  =  96
    exog    J   (previsions TSO)        E x 24  =  48   (E = 2)
    exog    J-1                         E x 24  =  48
    exog    J-7                         E x 24  =  48
    dummies jour de semaine                     =   7
                                                  ----
                                                   247   (le compte de Lago)

Selection de lambda : approche hybride de Lago (sec. 4.2.2). LARS avec un
critere d'information donne lambda, puis le modele est reestime par descente
de coordonnees. Quand la fenetre de calibration est plus courte que le nombre
de regresseurs, LassoLarsIC ne peut pas estimer la variance du bruit et leve
une ValueError : on passe alors par validation croisee.

Recalibration glissante, de grain reglable (recalib_every, en jours) :
  1   -> protocole Lago, ~721 reestimations
  7   -> compromis ; au-dela de 7 le gain est negligeable en pratique
  0   -> estimation unique sur le train

Ensemble : les previsions sont moyennees sur plusieurs fenetres de
calibration (calibration_windows, 0 = tout l'historique).

LEAR exige une entree complete : le masque est traite par apply_strategy(),
qui impute P_look avant la construction des features.
"""

import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from tqdm import tqdm

from models.basemodel import BaseForecaster

# decoupage d'un lookback de 168 h aligne sur les journees
DAY_SLICES = {1: slice(144, 168), 2: slice(120, 144),
              3: slice(96, 120), 7: slice(0, 24)}


def l2_normalize(X, stats=None):
    """Centrage puis division par la norme l2 de chaque colonne.

    Reproduit le normalize=True historique de LassoLarsIC (et NON une
    standardisation : la division est par la norme l2, pas par l'ecart-type).
    C'est ce qui rend lambda quasi independant du nombre d'echantillons.
    """
    if stats is None:
        mean = X.mean(axis=0)
        Xc = X - mean
        norm = np.linalg.norm(Xc, axis=0)
        norm[norm == 0] = 1.0
        stats = (mean, norm)
    else:
        mean, norm = stats
        Xc = X - mean
    return Xc / norm, stats


def fit_lasso_path(X, Y, criterion="bic", max_iter=2500):
    """24 LASSO, un par heure de livraison."""
    models, n_nonzero = [], []
    n, p = X.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for h in range(Y.shape[1]):
            if n > p:
                lars = LassoLarsIC(criterion=criterion, max_iter=max_iter)
                lars.fit(X, Y[:, h])
                alpha = lars.alpha_
            else:
                # n < p : LassoLarsIC ne peut pas estimer la variance du
                # bruit et leve une ValueError -- c'est l'erreur du ticket
                # #10 d'epftoolbox. On choisit lambda par validation croisee.
                cv = LassoCV(cv=5, max_iter=max_iter, n_jobs=-1)
                cv.fit(X, Y[:, h])
                alpha = cv.alpha_
            est = Lasso(alpha=alpha, max_iter=max_iter)
            est.fit(X, Y[:, h])
            models.append(est)
            n_nonzero.append(int((est.coef_ != 0).sum()))
    return models, n_nonzero


class Model(BaseForecaster):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg)
        self.name = "lear"
        c = cfg.model

        self.price_lags = list(c.price_lags)
        self.exog_lags = list(c.exog_lags)        # 0 = jour predit (X_fut)
        self.criterion = c.criterion
        self.max_iter = c.max_iter
        self.strategy = c.imputation
        self.recalib_every = int(c.recalib_every)
        self.use_val = bool(getattr(c, "use_val", True))
        self.use_dummies = bool(getattr(c, "use_dummies", True))
        self.use_lookback = bool(getattr(c, "use_lookback", True))
        self.calibration_windows = list(getattr(c, "calibration_windows", [0]))

        if self.lookback != 168:
            raise ValueError(
                f"LEAR attend un lookback de 168 h (7 jours), recu {self.lookback}"
            )

        # tous declares ici : nn.Module.__getattr__ leve une AttributeError
        # sur un attribut absent, y compris apres un rechargement partiel
        self.models = None            # {cw: (models, norm_stats)}
        self._recal = {}              # {(cw, jour_debut): (models, norm_stats)}
        self._recal_starts = None
        self._t0_test = None
        self._train_X = None
        self._train_Y = None
        self._train_t = None
        self.fitted = False

    # -- features -----------------------------------------------------------

    def _features(self, P, X_look, X_fut, t=None):
        feats = []

        # use_lookback=False : modele ablate, aucun prix passe en regresseur.
        # Borne inferieure de la tache sans historique.
        if self.use_lookback:
            feats += [P[:, DAY_SLICES[l]] for l in self.price_lags]

        for l in self.exog_lags:
            if l == 0:
                feats.append(X_fut.reshape(X_fut.shape[0], -1))
            elif self.use_lookback:
                feats.append(X_look[:, DAY_SLICES[l]].reshape(X_look.shape[0], -1))

        if self.use_dummies:
            # 7 dummies de jour de semaine. Le jour vient de l'indice temporel
            # absolu : avec stride 24, deux fenetres consecutives sont deux
            # jours consecutifs, donc t // 24 mod 7 les identifie de facon
            # coherente sans connaitre la date calendaire.
            day = (t // 24) % 7
            dow = np.zeros((len(day), 7), dtype=np.float32)
            dow[np.arange(len(day)), day] = 1.0
            feats.append(dow)

        return np.concatenate(feats, axis=1)

    def _batch_to_numpy(self, batch):
        from datasets.loader import apply_strategy
        P, _ = apply_strategy(batch["P_look"].cpu(), batch["mask"].cpu(),
                              self.strategy)
        return self._features(P.numpy(),
                              batch["X_look"].cpu().numpy(),
                              batch["X_fut"].cpu().numpy(),
                              batch["t"].cpu().numpy())

    def _collect(self, loader, desc):
        Xs, Ys, Ts = [], [], []
        for batch in tqdm(loader, desc=desc, leave=False):
            Xs.append(self._batch_to_numpy(batch))
            Ys.append(batch["Y"].cpu().numpy())
            Ts.append(batch["t"].cpu().numpy())
        return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ts)

    # -- estimation ---------------------------------------------------------

    def fit(self, loaders, device, logger=None):
        # LEAR n'a ni early stopping ni hyperparametre regle sur la
        # validation (lambda vient d'un critere in-sample), donc la validation
        # peut servir a la calibration. L'historique atteint ~1450 jours, soit
        # la plus longue fenetre de calibration de Lago.
        X, Y, T = self._collect(loaders["train_loader"], "LEAR features")
        if self.use_val and loaders.get("val_loader") is not None:
            Xv, Yv, Tv = self._collect(loaders["val_loader"], "LEAR val")
            Tv = Tv + T.max() + 24        # preserve l'ordre chronologique
            X, Y, T = (np.concatenate([X, Xv]), np.concatenate([Y, Yv]),
                       np.concatenate([T, Tv]))

        order = np.argsort(T)
        self._train_X, self._train_Y, self._train_t = X[order], Y[order], T[order]

        self.models = {}
        nz_all = []
        for cw in self.calibration_windows:
            m, s, nz = self._calibrate(self._train_X, self._train_Y, cw)
            self.models[cw] = (m, s)
            nz_all.extend(nz)
        self.fitted = True

        stats = {
            "n_features": float(X.shape[1]),
            "n_samples": float(X.shape[0]),
            "nonzero_mean": float(np.mean(nz_all)),
            "nonzero_min": float(np.min(nz_all)),
            "nonzero_max": float(np.max(nz_all)),
            "n_calibration_windows": float(len(self.calibration_windows)),
        }
        print(f"  design : {X.shape[0]} x {X.shape[1]}  |  fenetres : "
              f"{self.calibration_windows}  |  non nuls : "
              f"{stats['nonzero_mean']:.1f} ({stats['nonzero_min']:.0f}-"
              f"{stats['nonzero_max']:.0f})")
        if logger is not None:
            logger.log_metrics(stats, epoch=0, prefix="train")
        return {"train_loss": stats, "val_loss": {}}

    def _calibrate(self, X_hist, Y_hist, cw):
        """Estime sur les cw derniers jours (cw = 0 : tout l'historique)."""
        if cw > 0:
            X_hist, Y_hist = X_hist[-cw:], Y_hist[-cw:]
        Xn, stats = l2_normalize(X_hist)
        models, nz = fit_lasso_path(Xn, Y_hist, self.criterion, self.max_iter)
        return models, stats, nz

    # -- recalibration glissante -------------------------------------------

    def prepare_test(self, test_loader, logger=None):
        """Precalcule les jeux de coefficients pour la periode de test.

        A appeler avant forward_step quand recalib_every > 0. L'historique de
        chaque reestimation ne contient que du passe : pas de fuite.
        """
        if self.recalib_every <= 0:
            return
        if self._train_X is None:
            raise RuntimeError(
                "historique de calibration absent : appeler fit() d'abord, "
                "ou recharger un checkpoint qui le contient"
            )

        Xte, Yte, Tte = self._collect(test_loader, "LEAR test features")
        order = np.argsort(Tte)
        Xte, Yte, Tte = Xte[order], Yte[order], Tte[order]

        starts = list(range(0, len(Tte), self.recalib_every))
        X_all = np.concatenate([self._train_X, Xte])
        Y_all = np.concatenate([self._train_Y, Yte])
        n_train = len(self._train_X)

        self._recal = {}
        total = len(starts) * len(self.calibration_windows)
        with tqdm(total=total, desc="LEAR recalibration", leave=False) as bar:
            for s in starts:
                hist_end = n_train + s          # que du passe
                for cw in self.calibration_windows:
                    m, st, _ = self._calibrate(X_all[:hist_end],
                                               Y_all[:hist_end], cw)
                    self._recal[(cw, s)] = (m, st)
                    bar.update(1)

        self._recal_starts = np.array(starts)
        self._t0_test = int(Tte[0])
        if logger is not None:
            logger.log_metrics({"n_recalibrations": float(total)},
                               epoch=0, prefix="train")

    # -- inference ----------------------------------------------------------

    def _start_for(self, t):
        """Point de recalibration en vigueur pour la fenetre d'indice t."""
        if not self._recal or self._t0_test is None:
            return None
        day = (t - self._t0_test) // 24
        k = self._recal_starts[self._recal_starts <= day]
        return int(k[-1]) if len(k) else int(self._recal_starts[0])

    def _models_for(self, cw, start):
        if start is None:
            return self.models[cw]
        return self._recal[(cw, start)]

    def forward_step(self, batch, device):
        if not self.fitted:
            raise RuntimeError("LEAR non estime : appeler fit() d'abord")

        X = self._batch_to_numpy(batch)
        ts = batch["t"].cpu().numpy()
        usable = bool(self._recal) and self._t0_test is not None
        starts = np.array([self._start_for(int(t)) if usable else -1
                           for t in ts])

        # ensemble : moyenne des previsions sur les fenetres de calibration
        acc = np.zeros((X.shape[0], self.horizon), dtype=np.float64)
        for cw in self.calibration_windows:
            for s in np.unique(starts):
                sel = starts == s
                models, stats = self._models_for(cw, None if s == -1 else int(s))
                xn, _ = l2_normalize(X[sel], stats)
                for h in range(self.horizon):
                    acc[sel, h] += models[h].predict(xn)
        acc /= len(self.calibration_windows)

        return torch.as_tensor(acc, dtype=torch.float32, device=device)

    def configure_optimizer(self):
        return None                       # estimation en forme close

    # -- persistance --------------------------------------------------------

    def save(self, path):
        import joblib
        joblib.dump({
            "models": self.models,
            # _recal seul ne suffit pas : _recal_starts et _t0_test sont
            # indispensables pour retrouver le jeu de coefficients en vigueur
            "recal": self._recal,
            "recal_starts": self._recal_starts,
            "t0_test": self._t0_test,
            # l'historique permet a prepare_test de refaire des calibrations
            # apres rechargement (~1.4 Mo, negligeable)
            "train_X": self._train_X,
            "train_Y": self._train_Y,
            "train_t": self._train_t,
            "fitted": self.fitted,
            "spec": {"price_lags": self.price_lags,
                     "exog_lags": self.exog_lags,
                     "lookback": self.lookback, "horizon": self.horizon,
                     "strategy": self.strategy,
                     "recalib_every": self.recalib_every,
                     "calibration_windows": self.calibration_windows,
                     "use_dummies": self.use_dummies,
                     "use_lookback": self.use_lookback},
        }, path)

    @classmethod
    def load(cls, cfg, path, map_location="cpu"):
        import joblib
        m = cls(cfg)
        d = joblib.load(path)
        spec = d["spec"]

        # seul le schema des features doit correspondre : c'est lui qui
        # rendrait les predictions fausses sans erreur visible
        for k in ("price_lags", "exog_lags", "lookback", "horizon"):
            if spec.get(k) != getattr(m, k):
                raise ValueError(f"schema incompatible sur '{k}' : "
                                 f"{spec.get(k)} vs {getattr(m, k)}")

        m.models = d["models"]
        m.use_lookback = spec.get("use_lookback", m.use_lookback)   # <--
        m.use_dummies = spec.get("use_dummies", m.use_dummies)
        m._recal = d.get("recal", {})
        m._recal_starts = d.get("recal_starts")
        m._t0_test = d.get("t0_test")
        m._train_X = d.get("train_X")
        m._train_Y = d.get("train_Y")
        m._train_t = d.get("train_t")
        m.fitted = d["fitted"]

        # checkpoint anterieur : _recal a ete sauvegarde sans ses index
        # (_recal_starts, _t0_test), donc on ne peut pas savoir quel jeu de
        # coefficients s'applique a quelle fenetre. On retombe sur
        # l'estimation unique, qui elle est complete.
        if m._recal and (m._recal_starts is None or m._t0_test is None):
            print("      [LEAR] checkpoint anterieur : recalibration "
                  "glissante inexploitable, estimation unique utilisee")
            m._recal = {}

        return m