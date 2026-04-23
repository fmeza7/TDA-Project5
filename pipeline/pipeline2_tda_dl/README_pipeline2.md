# Pipeline 2 - Breakfast Actions (TDA + DL)

Esta carpeta contiene el flujo de `pipeline 2` para segmentacion temporal de acciones.

## 1. Flujo

1. `breakfast_manifest_builder.py`
2. `breakfast_cubical_preprocessing.py`
3. `breakfast_curves.py`
4. `build_frame_labels.py`
5. `breakfast_temporal_windows.py`
6. `train_breakfast_temporal_segmenter.py`
7. `infer_breakfast_temporal_segmenter.py`
8. `decode_breakfast_predictions.py`
9. `eval_breakfast_segmentation.py`

Notebook:

- `pipeline/pipeline2_tda_dl/experiments/breakfast_actions_playbook.ipynb`

## 2. Comando recomendado

Para correr todo el flujo desde PowerShell:

```powershell
cd "C:\ruta\al\repo\TDA-Project5"

powershell -ExecutionPolicy Bypass -File .\scripts\run_breakfast_pipeline2.ps1 `
  -VideosDir "C:\ruta\al\dataset\BreakfastII_15fps_qvga_sync\BreakfastII_15fps_qvga_sync" `
  -AnnotationsDir "C:\ruta\al\dataset\BreakfastII_15fps_qvga_sync\BreakfastII_15fps_qvga_sync" `
  -ArtifactsDir ".\pipeline\pipeline2_tda_dl\artifacts_breakfast_run1" `
  -SplitFile ".\pipeline\pipeline2_tda_dl\configs\splits_52_subjects.json"
```

Por defecto este script:

- usa solo videos en carpetas `cam01`
- toma las anotaciones como `frames`
- guarda la salida en la carpeta indicada en `-ArtifactsDir`

Archivos utiles al terminar:

- modelo: `pipeline/pipeline2_tda_dl/artifacts_breakfast_run1/temporal_model/breakfast_temporal_best.pt`
- evaluacion: `pipeline/pipeline2_tda_dl/artifacts_breakfast_run1/eval/eval_test.json`

## 3. Ejecucion por etapas

```bash
python -m pipeline.pipeline2_tda_dl.breakfast_manifest_builder \
  --videos_dir data/breakfast/videos \
  --annotations_dir data/breakfast/annotations \
  --split_file data/breakfast/splits.json \
  --output_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/manifest.json \
  --camera_dirs cam01

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
  --train_split_names train \
  --time_units frames

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
  --seed 1337 \
  --ignore_unknown \
  --class_weighting

python -m pipeline.pipeline2_tda_dl.infer_breakfast_temporal_segmenter \
  --windows_npz pipeline/pipeline2_tda_dl/artifacts_breakfast/windows/test_windows.npz \
  --model_checkpoint pipeline/pipeline2_tda_dl/artifacts_breakfast/temporal_model/breakfast_temporal_best.pt \
  --frame_labels_manifest pipeline/pipeline2_tda_dl/artifacts_breakfast/frame_labels/manifest_frame_labels.json \
  --output_dir pipeline/pipeline2_tda_dl/artifacts_breakfast/inference \
  --seed 1337

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

## 4. Split

Ejemplo minimo de `splits.json`:

```json
{
  "P03": "train",
  "P04": "train",
  "P16": "val",
  "P28": "test"
}
```

Split base para 52 sujetos:

- `pipeline/pipeline2_tda_dl/configs/splits_52_subjects.json`

## 5. Notas

- `label_map.json` se construye solo con labels de `train`.
- El manifest builder acepta anotaciones `.txt` y `.labels`.
- Si hay varias anotaciones para un mismo video, se toma la que corresponde a la misma vista del video, por ejemplo `cam01`.
- El parser de anotaciones soporta `start end label`, formato tabulado y rangos como `1-135 crack_egg`.
- Si las anotaciones estan en segundos, cambia `--time_units frames` por `--time_units seconds`.
- Algunos `.avi` pueden mostrar warnings del decodificador `ffmpeg`. Si el proceso sigue y los archivos de salida se generan bien, en general no debiesen impedir correr el pipeline.
