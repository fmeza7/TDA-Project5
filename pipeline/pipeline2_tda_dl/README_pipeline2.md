# Pipeline 2 – TDA → DL (Topological Autoencoder + Temporal Transformer)

Este pipeline complementario reutiliza los artefactos del pipeline original (1) y agrega una etapa profunda basada en embeddings topológicos y clasificación temporal. **No** modifica ningún archivo existente; todo vive dentro de `pipeline/pipeline2_tda_dl/`.

## Flujo completo

1. `window_dataset.py`: genera ventanas temporales de las curvas topológicas (`pipeline/feature_extraction/outputs_curves/`). Produce conjuntos para el TopoAE y para la etapa supervisada.
2. `train_topoae.py`: entrena un Topological Autoencoder (TopoAE) con pérdida de reconstrucción + pérdida topológica 0D basada en MST.
3. `export_latents.py`: infiere el TopoAE sobre todas las ventanas y guarda latentes alineados temporalmente.
4. `train_temporal_model.py`: construye secuencias de latentes y entrena un Transformer temporal con supervisión por clase (background + comerciales).
5. `infer_temporal_model.py`: infiere el modelo temporal sobre los latentes de TV, agrupa predicciones, aplica postprocesamiento y emite detecciones compatibles con `evaluar-v2.py`.
6. `evaluar-v2.py detecciones_tda_dl.txt gt.txt`: evalúa exactamente igual que el pipeline original.

## Entradas
- `pipeline/feature_extraction/outputs_curves/{tv,commercials}/*.npz`
- `pipeline/feature_extraction/outputs_curves/manifest_curves.json`
- `pipeline/preprocessing/outputs_cubical/manifest.json`
- `gt.txt`

## Artefactos principales
```
pipeline/pipeline2_tda_dl/artifacts/
  window_data/
    topoae_dataset.npz
    temporal_dataset.npz
  topoae/
    topoae_best.pt
    scaler_mean.npy
    scaler_std.npy
    topoae_config.json
  latents/
    tv/*.npz
    commercials/*.npz
    manifest_latents.json
  temporal_model/
    temporal_best.pt
    class_map.json
    temporal_config.json
  detections/
    detecciones_tda_dl.txt
```

## Comandos de referencia

```bash
python -m pipeline.pipeline2_tda_dl.window_dataset \
  --curves_dir pipeline/feature_extraction/outputs_curves \
  --preproc_manifest pipeline/preprocessing/outputs_cubical/manifest.json \
  --gt gt.txt \
  --output_dir pipeline/pipeline2_tda_dl/artifacts/window_data \
  --window_sec 8.0 \
  --stride_frames 1 \
  --positive_overlap 0.5 \
  --negative_overlap 0.1

python -m pipeline.pipeline2_tda_dl.train_topoae \
  --window_data pipeline/pipeline2_tda_dl/artifacts/window_data/topoae_dataset.npz \
  --output_dir pipeline/pipeline2_tda_dl/artifacts/topoae \
  --latent_dim 32 \
  --lambda_topo 0.5 \
  --batch_size 64 \
  --lr 1e-3 \
  --epochs 100

python -m pipeline.pipeline2_tda_dl.export_latents \
  --window_data pipeline/pipeline2_tda_dl/artifacts/window_data/temporal_dataset.npz \
  --topoae_dir pipeline/pipeline2_tda_dl/artifacts/topoae \
  --output_dir pipeline/pipeline2_tda_dl/artifacts/latents

python -m pipeline.pipeline2_tda_dl.train_temporal_model \
  --latents_dir pipeline/pipeline2_tda_dl/artifacts/latents \
  --output_dir pipeline/pipeline2_tda_dl/artifacts/temporal_model \
  --seq_len 9 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 50

python -m pipeline.pipeline2_tda_dl.infer_temporal_model \
  --latents_dir pipeline/pipeline2_tda_dl/artifacts/latents \
  --temporal_model_dir pipeline/pipeline2_tda_dl/artifacts/temporal_model \
  --preproc_manifest pipeline/preprocessing/outputs_cubical/manifest.json \
  --output pipeline/pipeline2_tda_dl/artifacts/detections/detecciones_tda_dl.txt \
  --score_threshold 0.6 \
  --merge_gap_sec 2.0 \
  --min_segment_sec 3.0

python evaluar-v2.py pipeline/pipeline2_tda_dl/artifacts/detecciones/detecciones_tda_dl.txt gt.txt
```

## Dependencias
- PyTorch >= 2.0
- NumPy, SciPy (para MST), scikit-learn
- tqdm / rich opcional para barras de progreso

> Nota: este pipeline es independiente del pipeline de k-NN original. Ambos pueden convivir y producir detecciones paralelas.
