# Pipeline TDA + k‑NN para detección de comerciales

Este directorio contiene la implementación modular del pipeline utilizado para detectar comerciales en las transmisiones de televisión usando topological data analysis (TDA) como señal de anomalía y búsqueda por similitud k‑NN para asignar cada ventana a un comercial específico. A continuación se describe cada etapa, los comandos principales y el mejor resultado obtenido hasta ahora.

## 1. Preprocesamiento (pipeline/preprocessing)

1. **Extracción cubical** (`preprocessing/cubical_preprocessing.py`):  
   - Se muestrean los videos de TV y comerciales a ~3 fps.  
   - Cada frame se normaliza (resize, escalado de intensidades) y se guarda como volumen para construir complejos cubicales.  
   - Salida: `pipeline/preprocessing/outputs_cubical/{tv|commercials}/*.npz` más un `manifest.json` con metadatos (duración, fps de muestreo, rutas).

2. **Outputs claves**:  
   - `pipeline/preprocessing/outputs_cubical/tv/mega-2014_04_10.npz` (etc).  
   - `pipeline/preprocessing/outputs_cubical/manifest.json`.

## 2. Extracción de características (pipeline/feature_extraction)

1. **Curvas topológicas** (`feature_extraction/topological_curves.py`):  
   - Se cargan los NPZ cubicales y se construyen diagramas de persistencia por frame.  
   - Se agregan estadísticas (conteos, suma/max/var de persistencias) y se genera la señal `combined_activity` que resume cambios morfológicos abruptos.  
   - Salida: `pipeline/feature_extraction/outputs_curves/{tv|commercials}/*_curves.npz` + `manifest_curves.json`.

2. **Similarity k‑NN** (`feature_extraction/knn_similarity.py` / `outputs_knn`):  
   - Para cada frame de TV se obtienen los k vecinos más cercanos entre todos los frames de comerciales usando embeddings de persistencia e histogramas (depende del script usado).  
   - Se guardan `timestamps`, índice del comercial, score y desfase estimado.  
   - Salida: `pipeline/feature_extraction/outputs_knn/tv/*_knn.npz` + `manifest_knn.json`.

## 3. Detección (pipeline/detection)

### 3.1 Detección por curvas (`detection/curve_detector.py`)
Se usa la señal `combined_activity` para detectar picos de actividad topológica y generar un archivo de hits basados solo en la curva. Útil como baseline de TDA puro.

### 3.2 Detección por k‑NN + curva (`detection/knn_detector.py`)
- El detector combina ambas fuentes:
  1. La curva topológica actúa como “gate”: sólo se abren pistas cuando el z-score (sin estandarizar por ahora) supera `--curve_threshold`.
  2. Dentro de cada ventana activa se toma el vecino dominante (voto por ventana de `--window_sec` segundos), se requiere score mínimo y se verifica que la pista cubra una fracción `coverage_ratio` del comercial.
  3. Las pistas se consolidan con tolerancia de desfase (`--offset_tolerance`) y se deduplican opcionalmente (`--min_gap`).
- Salida: `pipeline/detection/detecciones_knn.txt` listo para `evaluar-v2.py`.

### 3.3 Evaluación
Se usa `python evaluar-v2.py detecciones_knn.txt gt.txt` para obtener precisión, recall, F1, IoU y formato exigido por la tarea.

## 4. Comandos principales

1. **Curvas topológicas**
   ```bash
   python pipeline/feature_extraction/topological_curves.py \
     --input_dir pipeline/preprocessing/outputs_cubical \
     --output_dir pipeline/feature_extraction/outputs_curves \
     --smooth_window 5
   ```

2. **k‑NN detector (configuración que produjo el mejor resultado)**
   ```bash
   python pipeline/detection/knn_detector.py \
     --knn_dir pipeline/feature_extraction/outputs_knn \
     --manifest pipeline/preprocessing/outputs_cubical/manifest.json \
     --output pipeline/detection/detecciones_knn.txt \
     --score_threshold 0.28 \
     --offset_tolerance 2.0 \
     --coverage_ratio 0.08 \
     --min_frames 1 \
     --window_sec 0.80 \
     --curve_dir pipeline/feature_extraction/outputs_curves \
     --curve_threshold 1.5 \
     --include_score
   ```

3. **Evaluación**
   ```bash
   python evaluar-v2.py pipeline/detection/detecciones_knn.txt gt.txt
   ```

## 5. Resultado actual

Con la configuración anterior se obtuvo:

```
Precision = 89.0%
Recall    = 78.6%
F1        = 0.835
IoU       = 0.397
Correctas = 81
Incorrectas = 10
Repetidas = 61
Total_GT  = 103
Resultado Tarea = 68.9%
```

El aumento de recall se logró reduciendo `coverage_ratio` y `score_threshold` gracias al gating con la curva; la precisión se mantuvo aceptable pese a los falsos positivos, aunque aún hay margen para reducir `dp` (duplicadas) con estrategias como `min_gap` específico por comercial o ajustes de start_time más precisos.
