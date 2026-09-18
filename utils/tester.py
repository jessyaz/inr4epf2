import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from datasets.loader import inverse_price


def naive_seasonal(y_true):
    """Naive du protocole Lago : meme heure, meme jour de semaine, S-1.
    y_true : (N_jours, horizon), un jour par ligne (stride_eval = 24)."""
    return np.concatenate([y_true[:7], y_true[:-7]], axis=0)


def compute_metrics(y_hat, y_true, spike_q=0.90):
    err = np.abs(y_hat - y_true)
    mae = err.mean()
    naive = naive_seasonal(y_true)
    mae_naive = np.abs(naive - y_true).mean()

    # print("MAE naive :", mae_naive)
    # print("shape     :", y_true.shape)
    # naive_24 = np.concatenate([y_true[:1], y_true[:-1]], axis=0)
    # print("MAE naive J-1 :", np.abs(naive_24 - y_true).mean())
    # print("MAE naive J-7 :", mae_naive)

    thr = np.quantile(y_true, spike_q)
    spike = y_true >= thr

    denom = np.abs(y_hat) + np.abs(y_true)
    return {
        "MSE": float(((y_hat - y_true) ** 2).mean()),
        "RMSE": float(np.sqrt(((y_hat - y_true) ** 2).mean())),
        "MAE": float(mae),
        "rMAE": float(mae / mae_naive),
        "sMAPE": float(200 * np.mean(err / np.clip(denom, 1e-8, None))),
        "MAE_spike": float(err[spike].mean()),
        "spike_threshold": float(thr),
    }


@torch.no_grad()
def test(model, loader, scaler, device, logger=None, verbose=True):
    model.eval()

    preds, trues, masks = [], [], []
    for batch in tqdm(loader, desc="Testing", leave=False):
        pred = model.forward_step(batch, device)
        preds.append(pred.detach().cpu().numpy())
        trues.append(batch["Y"].numpy())
        masks.append(batch["mask"].numpy())

    preds = np.concatenate(preds, axis=0)        # (N, H)
    trues = np.concatenate(trues, axis=0)
    masks = np.concatenate(masks, axis=0).astype(bool)

    # metriques en EUR/MWh
    if scaler is not None:
        preds = inverse_price(scaler, preds)
        trues = inverse_price(scaler, trues)

    loss_dict = compute_metrics(preds, trues)
    eff = float((~masks).mean())
    loss_dict["masking_rate_effective"] = eff

    # pires et meilleures fenetres
    err = ((preds - trues) ** 2).mean(axis=1)
    ids = np.argsort(err)
    worst, best = ids[-3:], ids[:3]

    if logger is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        H = preds.shape[1]
        for ax, i in zip(axes.flatten(), list(worst) + list(best)):
            ax.plot(range(H), trues[i], marker="o", markersize=3, label="Target")
            ax.plot(range(H), preds[i], label="Pred")
            n_obs = int(masks[i].sum())
            ax.set_title(f"win {i} | {n_obs}/{masks.shape[1]} obs | "
                         f"mse {err[i]:.1f}", fontsize=8)
            ax.legend(fontsize=7)
        plt.tight_layout()
        logger.log_plot(fig, artifact_path="plot_test/test.png")
        plt.close(fig)
        logger.log_metrics(loss_dict, epoch=0, prefix="test")

    if verbose:
        for k, v in loss_dict.items():
            print(f"  {k:24s} {v:.4f}")

    return {"test_loss": loss_dict}