from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .temporal_dataset import build_sequence_records, summarize_records


def _summarize_npz(npz_path: Path) -> dict:
    if not npz_path.exists():
        return {"error": f"missing {npz_path}"}
    
    data = np.load(npz_path, allow_pickle=True)
    labels = data.get("label_id")
    videos = data.get("video_name")
    source_types = data.get("source_type")
    
    # En el Breakfast Dataset, labels puede ser una matriz de (N, seq_len), 
    # por lo que aplanamos para contar las clases correctamente.
    if labels is not None:
        labels = labels.flatten()
        
    summary = {
        "num_rows": int(data.get("z_latent", data.get("seq", [])).shape[0]) if "z_latent" in data or "seq" in data else 0,
        "labels": {},
        "videos": {},
        "source_types": {},
    }
    
    if labels is not None and len(labels) > 0:
        unique, counts = np.unique(labels, return_counts=True)
        summary["labels"] = {int(k): int(v) for k, v in zip(unique, counts)}
    
    # videos y source_types pueden ser arreglos o un solo string (según cómo se guardó)
    if videos is not None:
        if np.ndim(videos) == 0:
            summary["videos"] = {str(videos): summary["num_rows"]}
        else:
            unique, counts = np.unique(videos, return_counts=True)
            summary["videos"] = {str(k): int(v) for k, v in zip(unique, counts)}
            
    if source_types is not None:
        if np.ndim(source_types) == 0:
            summary["source_types"] = {str(source_types): summary["num_rows"]}
        else:
            unique, counts = np.unique(source_types, return_counts=True)
            summary["source_types"] = {str(k): int(v) for k, v in zip(unique, counts)}
            
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar datasets y latentes del pipeline 2 para Breakfast Dataset")
    parser.add_argument("--window_data_dir", type=Path, required=True)
    parser.add_argument("--latents_dir", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=15) # Ajustado al default lógico de acciones
    args = parser.parse_args()

    # 1. Resumen de los archivos de dataset consolidados
    topoae_npz = args.window_data_dir / "topoae_dataset.npz"
    temporal_npz = args.window_data_dir / "temporal_dataset.npz"
    
    print("[debug_dataset] topoae_dataset:", _summarize_npz(topoae_npz))
    print("[debug_dataset] temporal_dataset:", _summarize_npz(temporal_npz))

    # 2. Conteo de archivos latentes generados por TopoAE usando rglob (ignora carpetas)
    latents_files = sorted(args.latents_dir.rglob("*_latents.npz"))
    print(f"\n[debug_dataset] Total de archivos latentes encontrados: {len(latents_files)}")

    # 3. Prueba de carga del DataLoader Temporal
    # (El flag include_commercials fue eliminado, ahora lee dinámicamente todo)
    print("\n[debug_dataset] Construyendo records de secuencias temporales...")
    records = build_sequence_records(
        args.latents_dir, 
        args.seq_len, 
        min_label_id=0
    )
    print("[debug_dataset] Resumen de secuencias generadas:", summarize_records(records))


if __name__ == "__main__":
    main()