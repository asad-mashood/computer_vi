# Computer Vision Lab 06 Report: Landmark Recognition Using SIFT

**Author:** Muhammad Asad Mashood  
**Student ID:** FA23-BAI-039  
**Degree:** BS Artificial Intelligence  
**Institution:** FAST National University of Computer and Emerging Sciences  
**Course:** Computer Vision  
**Repository:** [ComputerVision-LABS](https://github.com/asad-mashood/ComputerVision-LABS)  
**Module:** `lab06/Landmark_Recognition_SIFT`  

---

## 1. Executive Summary

This report presents the implementation and empirical evaluation of an automated landmark recognition and planar pose localization system utilizing David Lowe's Scale-Invariant Feature Transform (SIFT). Designed for real-time tourism information centers, the architecture ingests unconstrained user captures under diverse environmental variations (scale variations, arbitrary planar rotations, aggressive illumination decreases, and significant viewpoint shifts) while rigorously filtering out non-database impostor images.

---

## 2. Theoretical Pipeline Architecture

```
[Query Image Ingestion]
        │
        ▼
[Color-to-Luminance Grayscale Conversion]
        │
        ▼
[SIFT Scale-Space Extrema Detection (DoG Pyramid)]
        │
        ▼
[Sub-Pixel Keypoint Refinement & Contrast/Edge Filtering]
        │
        ▼
[Orientation Assignment via Gradient Histograms]
        │
        ▼
[128-Dimensional Keypoint Descriptor Extraction]
        │
        ▼
[Brute-Force Matcher (Euclidean L2) + k-NN (k = 2)]
        │
        ▼
[Lowe's Distance Ratio Test (tau = 0.75)]
        │
        ▼
[Hypothesis Ranking & Automated Rejection Gate (T_match >= 20)]
        │
        ▼
[RANSAC Homography Estimation & Geometric Consistency Validation]
        │
        ▼
[Bounding Polygon Projection & Final Landmark Localization]
```

---

## 3. Experimental Benchmark Results

The system was evaluated against 6 representative operational conditions across the 5 reference landmarks and non-database queries:

| Condition | Ground Truth | Predicted Landmark | Query KPs | Matches | Inliers | Decision | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Landmark Image** | `eiffel_tower` | `eiffel_tower` | `2,946` | `2,946` | `2,946` | **RECOGNIZED** | `1368.0 ms` |
| **Rotated Landmark Image (45 deg)** | `eiffel_tower` | `eiffel_tower` | `2,517` | `1,234` | `1,227` | **RECOGNIZED** | `1131.8 ms` |
| **Resized Landmark Image (0.5x Scale)** | `eiffel_tower` | `eiffel_tower` | `672` | `509` | `508` | **RECOGNIZED** | `372.5 ms` |
| **Reduced Brightness Image (0.35x)** | `eiffel_tower` | `eiffel_tower` | `760` | `718` | `717` | **RECOGNIZED** | `392.5 ms` |
| **Different Viewpoint Image** | `eiffel_tower` | `eiffel_tower` | `1,020` | `1,020` | `1,020` | **RECOGNIZED** | `431.6 ms` |
| **Non-Database Landmark (Big Ben)** | `unknown` | `Unknown / Rejected` | `1,984` | `22` | `0` | **REJECTED** | `895.3 ms` |

---

## 4. Key Analytical Insights

1. **Scale and Rotation Invariance:** The canonical orientation assignment and multi-scale Gaussian octave representations allow SIFT to achieve near-perfect inlier ratios (> 90%) even under 45-degree rotation and 0.5x scaling.
2. **Illumination Invariance:** Luminance scaling (0.35x brightness reduction) retains 100% recognition accuracy because SIFT descriptors compute relative gradient orientations rather than absolute intensity values.
3. **Impostor Rejection:** The automated threshold gate successfully rejects unknown landmarks (Big Ben) with 0 false matches passing geometric verification, preventing tourism misinformation.
4. **Localization Robustness:** RANSAC geometric validation filters out non-convex or degenerate homographies, ensuring clean green polygon overlays for tourists.

---

## 5. Visual Artifacts

- **Reference Database:** `figures/task1_reference_database.png`
- **SIFT Extrema:** `figures/task2_sift_features.png`
- **Feature Matching Pairs:** `figures/task4_feature_matching_comparison.png`
- **Environmental Matrix:** `figures/task6_environmental_evaluation.png`
- **Telemetry Charts:** `figures/task6_benchmark_telemetry.png`
- **System Architecture:** `figures/pipeline_architecture.png`
