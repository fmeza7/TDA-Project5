from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .temporal_dataset import TemporalSequenceDataset, build_sequence_records, split_by_videos
from .temporal_model import TemporalClassifier


def _build_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    return torch.from_numpy(weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar modelo temporal Transformer")
    parser.add_argument("--latents_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_tv", nargs="*", default=[])
    parser.add_argument("--val_tv", nargs="*", default=[])
    args = parser.parse_args()

    records = build_sequence_records(args.latents_dir, args.seq_len, min_label_id=0)
    if not records:
        raise RuntimeError("No se generaron secuencias para entrenamiento")
    train_records, val_records = split_by_videos(records, args.train_tv, args.val_tv)

    train_dataset = TemporalSequenceDataset(train_records)
    val_dataset = TemporalSequenceDataset(val_records)
    latent_dim = train_records[0].seq.shape[-1]
    num_classes = int(max(rec.label_id for rec in records) + 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalClassifier(latent_dim=latent_dim, num_classes=num_classes, seq_len=args.seq_len).to(device)
    class_weights = _build_class_weights([rec.label_id for rec in train_records], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_weighted_f1": []}
    best_val = float("inf")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "temporal_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for seq, label in train_loader:
            seq = seq.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history["train_loss"].append(float(np.mean(losses)))

        model.eval()
        val_losses = []
        preds: List[int] = []
        targets: List[int] = []
        with torch.no_grad():
            for seq, label in val_loader:
                seq = seq.to(device)
                label = label.to(device)
                logits = model(seq)
                loss = criterion(logits, label)
                val_losses.append(loss.item())
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                targets.extend(label.cpu().tolist())
        avg_val = float(np.mean(val_losses))
        history["val_loss"].append(avg_val)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
        history["val_macro_f1"].append(macro_f1)
        history["val_weighted_f1"].append(weighted_f1)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({"state_dict": model.state_dict()}, model_path)

    config = vars(args)
    (output_dir / "temporal_config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    # save label names
    id_to_name = {rec.label_id: rec.label_name for rec in records if rec.label_name}
    class_map = {int(idx): id_to_name.get(idx, "__background__" if idx == 0 else f"class_{idx}") for idx in range(num_classes)}
    (output_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))


if __name__ == "__main__":
    main()
