# Topological Signatures for Broadcast Commercial Discovery

## 1. Title and Header  
**Topological Signatures for Broadcast Commercial Discovery in Chilean Prime-Time TV**  
Francisco Meza · francisco.meza@example.com (placeholder)  
Date: 2024-XX-XX  

## 2. Abstract  
We investigate whether commercial breaks embedded in full-length television (TV) recordings can be recovered reliably using only deterministic preprocessing plus a fusion of topological and k-NN similarity signals. The dataset comprises six evening broadcasts from Mega TV (April 2014) and twenty-one isolated commercials, all provided as MPEG/MP4 videos with a ground-truth file listing 103 annotated insertions. The pipeline remaps every video to 3 FPS, resizes frames to 224×224, extracts ResNet embeddings (2048 floats) and cubical-complex descriptors per frame, stores normalized NPZ files, then constructs sliding-window topological curves (combined Betti/H₁ activity). Stage 2 loads the preprocessed TV/commercial NPZs, computes k-nearest neighbors between every TV frame and commercial frames, and consolidates detections conditioned on the curve activity. Evaluations with the provided benchmark yield precision = 89.0 %, recall = 78.6 %, F1 = 0.835, IoU = 0.397, and task score 68.9 % (81 true positives, 10 false positives). Contributions include (i) a reproducible preprocessing recipe for letterboxed broadcast material, (ii) a hybrid curve+similarity detector with configurable gating, and (iii) a full evaluation harness compatible with the course metric. Limitations stem from limited nights of TV, lack of supervised learning over the TDA features, and persistent duplicates when commercials recur within a short gap.

## 3. Introduction  
Detecting commercials in long-form TV streams supports automated content auditing, advertiser reporting, and compliance monitoring. Conventional systems rely on heuristics (black frames, silence) or brute-force template matching on CNN embeddings; these can fail when broadcasters overlay graphics, change aspect ratios, or remix audio. We ask whether topological descriptors—capturing qualitative changes in the audio-visual embeddings—combined with lightweight similarity search can reliably spot known commercials. Persistent homology summarizes the connectivity and loop structures of sampled manifolds across scales, offering robustness to small perturbations. When commercials feature rapid cuts or distinctive jingles, their topological curves deviate sharply from the background programming. Objectives: (1) standardize all videos to comparable temporal/spatial resolution; (2) encode each frame/window as cubical-complex features plus ResNet embeddings; (3) gate k-NN detections with topological “activity” curves; (4) quantify performance with the course benchmark. We expect high precision for ads with distinctive topology (“Ferretería MTS,” “Dove Desodorante Hombre”) but challenges for conversational ads resembling regular programming.

## 4. Dataset Description  
**Source**: MP4/MPEG videos for six Mega TV broadcasts and twenty-one commercials, plus `gt.txt` listing 103 insertions.  
**Type**: synchronized audio-visual streams, ~25–30 FPS, 720p letterboxed.  
**Size**: each TV recording lasts ~55–60 minutes (10–11k frames after downsampling), commercials range 15–40 s.  
**Variables after preprocessing**: timestamps, frame indices, ResNet embeddings (2048), PCA projections (32), cubical persistence stats/images, manifest metadata (sampled FPS, duration).  
**Challenges**: letterboxing (bars causing low-variance regions), repeated commercials per night, overlapping jingles, and commercials that resemble the underlying program. Despite limited nights, `gt.txt` confirms 103 occurrences spanning 17 unique commercials.

## 5. Preprocessing  
1. **Temporal normalization**: sample every video at 3 FPS, ensuring uniform stride regardless of native FPS.  
2. **Spatial/feature normalization**: convert frames to RGB, apply ResNet transformations (resize, crop, channel normalization), and compute embeddings via ResNet50. In parallel, grayscale cubical volumes (48×48) feed persistence-diagram computation per frame.  
3. **Dimensionality reduction**: fit IncrementalPCA (32 components) on TV embeddings first, then transform both TV and commercial embeddings.  
4. **Packaging**: store NPZ files with embeddings, PCA, timestamps, frame indices, and persistence descriptors; log metadata in a manifest.  
**Justification**: downsampling avoids aliasing between commercials and regular segments, PCA mitigates high dimensionality before TDA, and persistence images/stats provide topological fingerprints for detectors that rely purely on per-frame topology.

## 6. TDA + Similarity Pipeline  
- **Topological curves**: For each TV NPZ, sliding windows (window_sec = 20, step_sec = 4) are summarized into combined activity curves mixing Betti counts, persistence landscapes, and brightness statistics; the signal `combined_activity` highlights windows with unusual topology.  
- **k-NN similarity**: For every TV frame, the nearest commercial frames are precomputed (storing commercial ID, timestamp, score).  
- **Detection logic**:  
  1. Iterate over TV frames in blocks of `window_sec`.  
  2. Use the curve to filter out windows whose activity z-score falls below `curve_threshold`.  
  3. Within active windows, vote the dominant commercial via the highest-count neighbor, average scores, ensure coverage (`coverage_ratio`), and enforce consistent offsets.  
  4. Optionally deduplicate detections if multiple hits of the same commercial occur within a short gap.  
- **Evaluation**: run `evaluar-v2.py detecciones_knn.txt gt.txt` to compute precision/recall/F1/IoU and sweep thresholds automatically.

## 7. Learning & Thresholding Strategy  
No supervised classifier is trained; the system acts as a deterministic similarity scanner with a tunable threshold over the scores produced by k-NN. The topological curve acts as a soft prior, reducing the effective search space. We experimented with coverage ratios, window sizes, and score thresholds. Deduplication (min_gap) was evaluated but ultimately left off for the best F1 because it reduced recall.

## 8. Results  
Best configuration:  
```
score_threshold    = 0.28
coverage_ratio     = 0.08
min_frames         = 1
window_sec         = 0.80
curve_threshold    = 1.5
offset_tolerance   = 2.0
```
Evaluated using the benchmark:  
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
Most true positives correspond to recurring commercials (“Ferretería MTS,” “Dove Pelo,” “Petrobras Pantuflas”), while remaining duplicates reflect repeated detections of the same ad in quick succession. False positives stem from short tracks that briefly resemble known commercials; raising coverage could reduce them at the cost of recall.

## 9. Discussion  
Topological gating proved useful: it allowed lowering the k-NN score threshold, significantly boosting recall without detonating precision. Resampling and consistent PCA basis minimized biases between TV nights. Limitations: (i) duplicates remain high, suggesting the need for temporal reasoning or min-gap per commercial; (ii) the curve currently uses raw activity values—standardizing them could make thresholding more interpretable; (iii) combination with supervised classifiers could exploit the rich cubical features more effectively. Nonetheless, the pipeline demonstrates that persistent structures and lightweight similarity can deliver actionable detections on real broadcast data.

## 10. Conclusion & Next Steps  
We produced a reproducible, hybrid TDA + k-NN pipeline that achieves F1 ≈ 0.835 on the benchmark. Future work includes:  
1. Normalizing the curve signals (z-score) to ensure consistent gating across broadcasts.  
2. Implementing adaptive coverage requirements per commercial duration to reduce false positives.  
3. Introducing supervised models (SVM or lightweight CNN) trained on the TDA vectors.  
4. Adding better deduplication logic (min-gap per brand, start-time refinement via offset averaging).  

## 11. Reproducibility  
Run the stages sequentially: preprocessing → curve extraction → k-NN similarity → detection → evaluation. Environment: Python 3.10, PyTorch 2.1, torchvision 0.16, NumPy 1.26, SciPy 1.11, scikit-learn 1.3, librosa 0.10, ripser 0.6, persim 0.3, gudhi 3.8, OpenCV 4.8. All scripts are deterministic given fixed manifests; random seeds rely on NumPy defaults.
