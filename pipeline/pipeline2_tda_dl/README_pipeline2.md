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

## Adaptacion Breakfast (action units frame-level)

La adaptacion a Breakfast mantiene la parte TDA del repo y reemplaza el bloque k-NN por supervision temporal frame-level.

### Flujo recomendado

1. `breakfast_manifest_builder.py`: crea un manifest con `video_id`, `subject_id`, `activity_label`, `split`, `annotation_path`.
2. `breakfast_cubical_preprocessing.py`: extrae `tda_features` y `timestamps_sec` por video usando el motor cubical existente.
3. `breakfast_curves.py`: genera `curve_signals` por video/split.
4. `build_frame_labels.py`: alinea anotaciones temporales a `timestamps_sec` y crea `frame_label_ids`.
5. `breakfast_temporal_windows.py`: empaqueta ventanas many-to-many (`X`, `y`, `valid_mask`) para entrenamiento temporal.
6. `train_breakfast_temporal_segmenter.py`: entrena un BiLSTM many-to-many sobre las ventanas.
7. `infer_breakfast_temporal_segmenter.py`: reconstruye predicciones frame-level por video (promedio de logits en ventanas solapadas).
8. `decode_breakfast_predictions.py`: suaviza y limpia la secuencia temporal (mode filter + merge de segmentos cortos).
9. `eval_breakfast_segmentation.py`: evalua `frame_accuracy`, `edit_score`, `F1@10/25/50`.

### Comandos base

```bash
python -m pipeline.pipeline2_tda_dl.breakfast_manifest_builder \
  --videos_dir data/breakfast/videos \
  --annotations_dir data/breakfast/annotations \
  --output_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/manifest.json

python -m pipeline.pipeline2_tda_dl.breakfast_cubical_preprocessing \
  --dataset_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/manifest.json \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/outputs_cubical \
  --sample_fps 3.0

python -m pipeline.pipeline2_tda_dl.breakfast_curves \
  --input_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/outputs_cubical \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/outputs_curves \
  --smooth_window 5

python -m pipeline.pipeline2_tda_dl.build_frame_labels \
  --dataset_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/manifest.json \
  --cubical_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/outputs_cubical/manifest_cubical.json \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/frame_labels \
  --train_split_names train

python -m pipeline.pipeline2_tda_dl.breakfast_temporal_windows \
  --cubical_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/outputs_cubical/manifest_cubical.json \
  --curves_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/outputs_curves/manifest_curves.json \
  --frame_labels_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/frame_labels/manifest_frame_labels.json \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/windows \
  --window_size 31 \
  --stride_train 5 \
  --stride_val 5 \
  --stride_test 1

python -m pipeline.pipeline2_tda_dl.train_breakfast_temporal_segmenter \
  --windows_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/windows \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/temporal_model \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3 \
  --ignore_unknown \
  --class_weighting

python -m pipeline.pipeline2_tda_dl.infer_breakfast_temporal_segmenter \
  --windows_npz pipeline/pipeline2_tda_dl/artifacts_breakfast/windows/test_windows.npz \
  --model_checkpoint pipeline/pipeline2_tda_dl/artifacts_breakfast/temporal_model/breakfast_temporal_best.pt \
  --frame_labels_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/frame_labels/manifest_frame_labels.json \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/inference

python -m pipeline.pipeline2_tda_dl.decode_breakfast_predictions \
  --raw_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/inference/raw_predictions_manifest.json \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/decoded \
  --kernel_size 5 \
  --min_segment_sec 0.5

python -m pipeline.pipeline2_tda_dl.eval_breakfast_segmentation \
  --decoded_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/decoded/decoded_manifest.json \
  --frame_labels_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/frame_labels/manifest_frame_labels.json \
  --splits test \
  --output_json pipeline/pipeline2_tda_dl/artifacts_breakfast/eval/eval_test.json
```

Notas de protocolo:
- `label_map.json` se construye solo con labels de `train` para evitar leakage de test.
- El split debe ser por `subject_id` (no por ventanas).
- `valid_mask` se guarda desde el inicio para soportar padding/batches variables en la etapa de modelado temporal.
