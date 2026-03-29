from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from .frame_dataset import build_dataloaders
from .io_utils import ensure_dir, resolve_device
from .visual_autoencoder import VisualAutoencoder


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for batch, _ in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(sum(losses) / len(losses))


def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            losses.append(loss.item())
    return float(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar autoencoder visual para pipeline 3")
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device()
    print(f"[visual_ae] usando device={device}")
    train_loader, val_loader = build_dataloaders(args.frames_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = VisualAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    history: Dict[str, list] = {"train": [], "val": []}
    best_val = float("inf")
    output_dir = args.output_dir
    ensure_dir(output_dir)
    ckpt_path = output_dir / "visual_ae_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"[visual_ae] epoch={epoch}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "latent_dim": args.latent_dim}, ckpt_path)

    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (output_dir / "visual_ae_config.json").write_text(json.dumps(config, indent=2))
    print(f"[visual_ae] entrenamiento finalizado; mejor val={best_val:.5f}")


if __name__ == "__main__":
    main()
