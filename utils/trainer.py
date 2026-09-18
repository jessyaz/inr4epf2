from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from utils.valider import validate


def _plot_batch(model, batch, pred, device, n_plots=6):
    """Lookback (avec zones masquees) + horizon predit vs cible."""
    P = batch["P_look"].detach().cpu().numpy()
    M = batch["mask"].detach().cpu().numpy().astype(bool)
    Y = batch["Y"].detach().cpu().numpy()
    Pr = pred.detach().cpu().numpy()

    L, H = model.lookback, model.horizon
    n_plots = min(n_plots, P.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flatten()[:n_plots]):
        # lookback : trous en NaN pour qu'ils ne soient pas traces
        past = P[i].copy()
        past[~M[i]] = float("nan")
        ax.plot(range(L), past, color="tab:gray", lw=1, label="Lookback")

        ax.plot(range(L, L + H), Y[i], marker="o", markersize=3, label="Target")
        ax.plot(range(L, L + H), Pr[i], label="Pred")
        ax.axvline(L - 0.5, color="k", lw=0.8, ls="--")

        # zones masquees du lookback
        start, inside = None, False
        for j, valid in enumerate(M[i]):
            if not valid and not inside:
                start, inside = j, True
            elif valid and inside:
                ax.axvspan(start - 0.5, j - 0.5, color="red", alpha=0.12)
                inside = False
        if inside:
            ax.axvspan(start - 0.5, L - 0.5, color="red", alpha=0.12)

        ax.set_title(f"sample {i}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def train(model, loaders, optimizer, device, logger=None):
    train_loader, val_loader = loaders["train_loader"], loaders["val_loader"]

    num_epochs = model.cfg.train.num_epochs
    patience = model.cfg.train.patience
    plot_every = getattr(model.cfg.train, "plot_every", 0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, cooldown=3, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    val_loss_dict = {"MSE": float("nan")}

    for epoch in range(num_epochs):
        model.train()
        model.set_epoch(epoch)

        running, n = 0.0, 0
        for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)):

            pred = model.forward_step(batch, device)
            target = batch["Y"].to(device)
            loss = ((pred - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            n += 1

            if (plot_every and logger is not None
                    and batch_idx == 5 and (epoch + 1) % plot_every == 0):
                fig = _plot_batch(model, batch, pred, device)
                logger.log_plot(fig, artifact_path=f"plots/epoch_{epoch + 1}.png")
                plt.close(fig)

        loss_dict = {"MSE": running / max(n, 1)}
        val_loss_dict = validate(model, val_loader, device)["val_loss"]

        if logger is not None:
            logger.log_metrics(loss_dict, epoch=epoch, prefix="train")
            logger.log_metrics(val_loss_dict, epoch=epoch, prefix="val")

        current = val_loss_dict["MSE"]
        scheduler.step(current)
        print(f"epoch {epoch + 1:3d}  train {loss_dict['MSE']:.5f}  val {current:.5f}")

        if current < best_val_loss:
            best_val_loss = current
            best_state_dict = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping epoch {epoch + 1} (best val {best_val_loss:.6f})")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {"train_loss": loss_dict, "val_loss": val_loss_dict}