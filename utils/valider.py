import torch


@torch.no_grad()
def validate(model, val_loader, device):
    """MSE moyenne en espace scale. Val propre (rate=0)."""
    model.eval()
    total, n = 0.0, 0

    for batch in val_loader:
        pred = model.forward_step(batch, device)
        target = batch["Y"].to(device)
        total += ((pred - target) ** 2).mean().item()
        n += 1

    return {"val_loss": {"MSE": total / max(n, 1)}}