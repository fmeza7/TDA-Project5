from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .temporal_dataset import (
    SequenceRecord,
    TemporalSequenceDataset,
    build_sequence_records,
    split_by_videos,
    summarize_records,
)
from .temporal_model import TemporalTransformer # Se asume el nombre actualizado de la clase


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_class_weights(labels_list: List[np.ndarray], num_classes: int) -> torch.Tensor:
    # Como labels_list es una lista de arreglos (una etiqueta por frame), los aplanamos
    flat_labels = np.concatenate(labels_list)
    counts = np.bincount(flat_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    return torch.from_numpy(weights)


def oversample_minority_records(records: List["SequenceRecord"], min_positive_per_class: int = 500):
    from collections import defaultdict

    by_label = defaultdict(list)
    for rec in records:
        # Usamos el frame central para definir la "clase principal" de la ventana para el sobremuestreo
        center_idx = len(rec.label_id) // 2
        primary_label = int(rec.label_id[center_idx])
        by_label[primary_label].append(rec)

    new_records = list(records)
    for label_id, items in by_label.items():
        if label_id == 0:  # No sobremuestrear el fondo
            continue
        if 0 < len(items) < min_positive_per_class:
            extra = min_positive_per_class - len(items)
            new_records.extend(random.choices(items, k=extra))
    return new_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar modelo temporal Transformer para segmentación densa")
    parser.add_argument("--latents_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=15) # Ajustado a un valor más lógico para acciones
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_tv", nargs="*", default=[])
    parser.add_argument("--val_tv", nargs="*", default=[])
    parser.add_argument("--include_commercials", action="store_true")
    args = parser.parse_args()

    device = resolve_device()
    print(f"Using device: {device}")

    records = build_sequence_records(
        args.latents_dir,
        args.seq_len,
        min_label_id=0,
        include_commercials=args.include_commercials,
    )
    print(f"[temporal] total sequence records: {len(records)}")
    if records:
        unique_videos = sorted({rec.video_name for rec in records})
        print(f"[temporal] videos present: {unique_videos}")
    if not records:
        raise RuntimeError("No se generaron secuencias para entrenamiento")
        
    train_records, val_records = split_by_videos(records, args.train_tv, args.val_tv)
    print("[temporal] train summary:", summarize_records(train_records))
    print("[temporal] val summary:", summarize_records(val_records))
    
    train_records = oversample_minority_records(train_records, min_positive_per_class=500)
    print("[temporal] train summary after oversampling:", summarize_records(train_records))

    if len(train_records) == 0:
        raise RuntimeError("No hay secuencias de entrenamiento")
    if len(val_records) == 0:
        raise RuntimeError("No hay secuencias de validación")

    train_dataset = TemporalSequenceDataset(train_records)
    val_dataset = TemporalSequenceDataset(val_records)
    latent_dim = train_records[0].seq.shape[-1]
    
    # max() iterando sobre el np.max de cada arreglo de labels
    num_classes = int(max(np.max(rec.label_id) for rec in records) + 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TemporalTransformer(input_dim=latent_dim, num_classes=num_classes).to(device)
    
    class_weights = _build_class_weights([rec.label_id for rec in train_records], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_weighted_f1": [], "val_accuracy": []}
    best_val = float("inf")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "temporal_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for seq, label in train_loader:
            seq = seq.to(device)
            label = label.to(device) # shape: (batch_size, seq_len)
            
            optimizer.zero_grad()
            logits = model(seq) # shape: (batch_size, seq_len, num_classes)
            
            # Transponer para CrossEntropyLoss: (batch_size, num_classes, seq_len)
            logits_ce = logits.transpose(1, 2)
            
            loss = criterion(logits_ce, label)
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
                logits_ce = logits.transpose(1, 2)
                
                loss = criterion(logits_ce, label)
                val_losses.append(loss.item())
                
                # Aplanar las predicciones y targets para calcular métricas frame a frame
                pred_frames = torch.argmax(logits, dim=2).flatten().cpu().tolist()
                target_frames = label.flatten().cpu().tolist()
                
                preds.extend(pred_frames)
                targets.extend(target_frames)
                
        avg_val = float(np.mean(val_losses))
        history["val_loss"].append(avg_val)
        
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
        history["val_macro_f1"].append(macro_f1)
        history["val_weighted_f1"].append(weighted_f1)
        
        acc = accuracy_score(targets, preds)
        history["val_accuracy"].append(acc)

        if np.isnan(avg_val):
            raise RuntimeError("La validación produjo NaN; revise el split o los datos.")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_val": best_val,
                    "seq_len": args.seq_len,
                    "latent_dim": latent_dim,
                    "num_classes": num_classes,
                },
                model_path,
            )
            print(f"[temporal] epoch={epoch} new best val_loss={avg_val:.6f} -> saved {model_path}")

        print(
            f"[temporal] epoch={epoch}/{args.epochs} "
            f"train_loss={history['train_loss'][-1]:.6f} "
            f"val_loss={avg_val:.6f} "
            f"val_acc(MoF)={acc:.4f} "
            f"val_macro_f1={macro_f1:.4f} "
            f"val_weighted_f1={weighted_f1:.4f}"
        )

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (output_dir / "temporal_config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    
    # Construir mapa de nombres de clase iterando sobre las secuencias
    id_to_name = {}
    for rec in records:
        for lbl_id, lbl_name in zip(rec.label_id, rec.label_name):
            if lbl_name:
                id_to_name[int(lbl_id)] = lbl_name
                
    class_map = {int(idx): id_to_name.get(idx, "__background__" if idx == 0 else f"class_{idx}") for idx in range(num_classes)}
    (output_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))
    
    split_summary = {
        "train": summarize_records(train_records),
        "val": summarize_records(val_records),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2))


if __name__ == "__main__":
    main()