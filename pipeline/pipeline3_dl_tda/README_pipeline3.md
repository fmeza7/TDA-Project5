# Pipeline 3 — DL → TDA

Este pipeline agrega una tercera ruta experimental que primero aprende **embeddings visuales** por frame con un autoencoder simple y, sobre esas representaciones, vuelve a aplicar **TDA** para generar curvas y detecciones similares al pipeline clásico. Todo el código vive en `pipeline/pipeline3_dl_tda/` y no modifica los pipelines previos.

## Flujo completo

1. `frame_sampling.py`: muestrea frames de TV y comerciales (`.npz` con las secuencias normalizadas).
2. `train_visual_autoencoder.py`: entrena un autoencoder convolucional liviano sobre los frames.
3. `export_frame_embeddings.py`: aplica el encoder entrenado y guarda embeddings por frame.
4. `embedding_tda.py`: calcula TDA ventana por ventana sobre la secuencia de embeddings.
5. `curve_builder.py`: empaqueta los features topológicos como curvas 1D.
6. `knn_on_embedding_tda.py`: arma un banco k-NN con los commercials y matchea cada ventana de TV.
7. `detector_embedding_tda.py`: usa la curva `combined_activity_z` como gate y genera detecciones compatibles con `evaluar-v2.py`.
8. `python evaluar-v2.py pipeline/pipeline3_dl_tda/artifacts/detections/detecciones_dl_tda.txt gt.txt`: evalúa igual que los otros pipelines.

## Comandos recomendados

```bash
python -m pipeline.pipeline3_dl_tda.frame_sampling \
  --tv_dir data/television \
  --commercials_dir data/comerciales \
  --output_dir pipeline/pipeline3_dl_tda/artifacts/frames \
  --sample_fps 3.0 \
  --image_size 48

python -m pipeline.pipeline3_dl_tda.train_visual_autoencoder \
  --frames_dir pipeline/pipeline3_dl_tda/artifacts/frames \
  --output_dir pipeline/pipeline3_dl_tda/artifacts/visual_ae \
  --latent_dim 64 \
  --batch_size 128 \
  --lr 1e-3 \
  --epochs 30

python -m pipeline.pipeline3_dl_tda.export_frame_embeddings \
  --frames_dir pipeline/pipeline3_dl_tda/artifacts/frames \
  --visual_ae_dir pipeline/pipeline3_dl_tda/artifacts/visual_ae \
  --output_dir pipeline/pipeline3_dl_tda/artifacts/frame_embeddings

python -m pipeline.pipeline3_dl_tda.embedding_tda \
  --embeddings_dir pipeline/pipeline3_dl_tda/artifacts/frame_embeddings \
  --output_dir pipeline/pipeline3_dl_tda/artifacts/embedding_tda \
  --window_frames 12 \
  --stride 1

python -m pipeline.pipeline3_dl_tda.curve_builder \
  --tda_dir pipeline/pipeline3_dl_tda/artifacts/embedding_tda \
  --output_dir pipeline/pipeline3_dl_tda/artifacts/embedding_curves

python -m pipeline.pipeline3_dl_tda.knn_on_embedding_tda \
  --tda_dir pipeline/pipeline3_dl_tda/artifacts/embedding_tda \
  --output_dir pipeline/pipeline3_dl_tda/artifacts/embedding_knn \
  --k 5 \
  --normalize

python -m pipeline.pipeline3_dl_tda.detector_embedding_tda \
  --knn_dir pipeline/pipeline3_dl_tda/artifacts/embedding_knn \
  --curve_dir pipeline/pipeline3_dl_tda/artifacts/embedding_curves \
  --preproc_manifest pipeline/preprocessing/outputs_cubical/manifest.json \
  --output pipeline/pipeline3_dl_tda/artifacts/detections/detecciones_dl_tda.txt \
  --score_threshold 0.5 \
  --curve_threshold 1.5 \
  --merge_gap_sec 2.0 \
  --min_segment_sec 3.0

python evaluar-v2.py pipeline/pipeline3_dl_tda/artifacts/detecciones/detecciones_dl_tda.txt gt.txt
```
