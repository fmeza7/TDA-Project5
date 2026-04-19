from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .breakfast_dataset import load_window_npz, make_dataloader, summarize_windows
from .breakfast_temporal_segmenter import TDATemporalSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenar segmentador temporal many-to-many para Breakfast"
    )
    parser.add_argument("--windows_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_file", type=str, default="train_windows.npz")
    parser.add_argument("--val_file", type=str, default="val_windows.npz")
    parser.add_argument("--label_map", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ignore_unknown", action="store_true")
    parser.add_argument("--class_weighting", action="store_true")
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_label_map(path: Path | None, windows_dir: Path) -> Dict[str, int]:
    candidates: List[Path] = []
    if path is not None:
        candidates.append(path)
    candidates.append(windows_dir / "label_map.json")
    candidates.append(windows_dir.parent / "frame_labels" / "label_map.json")

    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): int(v) for k, v in payload.items()}
    return {}


def _infer_num_classes(
    train_y: np.ndarray, val_y: np.ndarray, label_map: Dict[str, int]
) -> int:
    candidates = [0]
    if train_y.size:
        candidates.append(int(train_y.max()))
    if val_y.size:
        candidates.append(int(val_y.max()))
    if label_map:
        candidates.append(max(label_map.values()))
    return int(max(candidates) + 1)


def _masked_targets(
    y: torch.Tensor,
    mask: torch.Tensor,
    ignore_unknown: bool,
    unknown_id: int | None,
) -> torch.Tensor:
    valid = mask.reshape(-1) > 0
    if ignore_unknown and unknown_id is not None:
        valid = valid & (y.reshape(-1) != unknown_id)
    return valid


def _compute_class_weights(
    y: np.ndarray,
    valid_mask: np.ndarray,
    num_classes: int,
    ignore_unknown: bool,
    unknown_id: int | None,
) -> torch.Tensor:
    flat_y = y.reshape(-1)
    flat_valid = valid_mask.reshape(-1) > 0
    if ignore_unknown and unknown_id is not None:
        flat_valid &= flat_y != unknown_id
    effective = flat_y[flat_valid]
    if effective.size == 0:
        return torch.ones((num_classes,), dtype=torch.float32)

    counts = np.bincount(effective, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    return torch.from_numpy(weights)


def _evaluate(
    model: TDATemporalSegmenter,
    loader,
    device: torch.device,
    num_classes: int,
    class_weights: torch.Tensor | None,
    ignore_unknown: bool,
    unknown_id: int | None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    preds_all: List[int] = []
    target_all: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["valid_mask"].to(device)

            logits = model(x)
            flat_logits = logits.reshape(-1, num_classes)
            flat_y = y.reshape(-1)
            valid = _masked_targets(flat_y, mask, ignore_unknown, unknown_id)
            if not torch.any(valid):
                continue

            step_logits = flat_logits[valid]
            step_targets = flat_y[valid]
            loss = F.cross_entropy(step_logits, step_targets, weight=class_weights)
            total_loss += float(loss.item())
            total_steps += 1

            step_preds = torch.argmax(step_logits, dim=1)
            preds_all.extend(step_preds.cpu().tolist())
            target_all.extend(step_targets.cpu().tolist())

    if not target_all:
        return {"loss": float("inf"), "frame_acc": 0.0, "macro_f1": 0.0}

    preds_np = np.asarray(preds_all)
    targets_np = np.asarray(target_all)
    frame_acc = float((preds_np == targets_np).mean())
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
    avg_loss = total_loss / max(1, total_steps)
    return {"loss": float(avg_loss), "frame_acc": frame_acc, "macro_f1": macro_f1}


def main() -> None:
    args = parse_args()
    device = resolve_device()
    print(f"Using device: {device}")

    train_path = args.windows_dir / args.train_file
    val_path = args.windows_dir / args.val_file
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"No existen windows train/val: {train_path}, {val_path}"
        )

    train_np = load_window_npz(train_path)
    val_np = load_window_npz(val_path)
    label_map = _load_label_map(args.label_map, args.windows_dir)
    unknown_id = label_map.get("__unk__")
    num_classes = _infer_num_classes(train_np.y, val_np.y, label_map)

    input_dim = int(train_np.X.shape[-1])
    model = TDATemporalSegmenter(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    train_loader = make_dataloader(
        train_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        val_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    class_weights = None
    if args.class_weighting:
        class_weights = _compute_class_weights(
            y=train_np.y,
            valid_mask=train_np.valid_mask,
            num_classes=num_classes,
            ignore_unknown=args.ignore_unknown,
            unknown_id=unknown_id,
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "breakfast_temporal_best.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_frame_acc": [],
        "val_macro_f1": [],
    }
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["valid_mask"].to(device)

            logits = model(x)
            flat_logits = logits.reshape(-1, num_classes)
            flat_y = y.reshape(-1)
            valid = _masked_targets(flat_y, mask, args.ignore_unknown, unknown_id)
            if not torch.any(valid):
                continue

            loss = F.cross_entropy(
                flat_logits[valid], flat_y[valid], weight=class_weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            class_weights=class_weights,
            ignore_unknown=args.ignore_unknown,
            unknown_id=unknown_id,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_frame_acc"].append(val_metrics["frame_acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "best_val_loss": best_val,
                    "label_map": label_map,
                },
                best_path,
            )

        print(
            f"[train_breakfast_temporal_segmenter] epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['frame_acc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (output_dir / "train_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "train_windows_summary.json").write_text(
        json.dumps(summarize_windows(train_path), indent=2),
        encoding="utf-8",
    )
    (output_dir / "val_windows_summary.json").write_text(
        json.dumps(summarize_windows(val_path), indent=2),
        encoding="utf-8",
    )

    print(f"[train_breakfast_temporal_segmenter] best_checkpoint={best_path}")


if __name__ == "__main__":
    main()
