from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .topo_loss import topo_loss_0d
from .topoae_model import TopoAutoencoder


def _select_indices(
    video_names: np.ndarray,
    source_types: np.ndarray,
    train_tv: Sequence[str],
    val_tv: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    tv_mask = source_types == "tv"
    commercials_mask = source_types == "commercial"
    unique_tv = sorted(set(video_names[tv_mask]))
    train_set = set(train_tv) if train_tv else set(unique_tv[: max(1, len(unique_tv) - 1)])
    val_set = set(val_tv) if val_tv else set(unique_tv) - train_set
    if not val_set:
        val_candidates = [tv for tv in unique_tv if tv not in train_set]
        if val_candidates:
            val_set.add(val_candidates[0])
    train_idx = np.where((tv_mask & np.isin(video_names, list(train_set))) | commercials_mask)[0]
    val_idx = np.where(tv_mask & np.isin(video_names, list(val_set)))[0]
    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError("Train/val split vacío; revise --train_tv y --val_tv")
    return train_idx, val_idx


class WindowDataset(Dataset):
    def __init__(self, tensors: np.ndarray):
        self.tensors = tensors.astype(np.float32)

    def __len__(self) -> int:
        return self.tensors.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.tensors[idx])


def train_loop(
    model: TopoAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lambda_topo: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history = {"train": [], "val": []}
    best_val = float("inf")
    epochs_no_improve = 0
    model_path = output_dir / "topoae_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            recon_loss = criterion(recon, batch)
            topo_loss, _, _ = topo_loss_0d(batch, model.encode(batch))
            loss = recon_loss + lambda_topo * topo_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = float(np.mean(train_losses))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                recon_loss = criterion(recon, batch)
                topo_loss, _, _ = topo_loss_0d(batch, model.encode(batch))
                val_losses.append((recon_loss + lambda_topo * topo_loss).item())
        avg_val = float(np.mean(val_losses))
        history["train"].append(avg_train)
        history["val"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict()}, model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento TopoAE")
    parser.add_argument("--window_data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--lambda_topo", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_tv", nargs="*", default=[])
    parser.add_argument("--val_tv", nargs="*", default=[])
    args = parser.parse_args()

    data = np.load(args.window_data, allow_pickle=True)
    X = data["X_flat"]
    video_names = data["video_name"]
    source_types = data["source_type"]
    train_idx, val_idx = _select_indices(video_names, source_types, args.train_tv, args.val_tv)

    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0) + 1e-6
    X_train = (X[train_idx] - mean) / std
    X_val = (X[val_idx] - mean) / std

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "scaler_mean.npy", mean)
    np.save(output_dir / "scaler_std.npy", std)

    train_loader = DataLoader(WindowDataset(X_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopoAutoencoder(input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)

    history = train_loop(
        model,
        train_loader,
        val_loader,
        args.lambda_topo,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        device,
        output_dir,
    )

    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    config = vars(args)
    (output_dir / "topoae_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
