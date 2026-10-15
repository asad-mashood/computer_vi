# Computer Vision Lab 06: Landmark Recognition Using SIFT

> [!NOTE]  
> **Coursework Repository Migration:**  
> This laboratory module is actively maintained as part of the centralized coursework portfolio at:  
> 👉 **[ComputerVision-LABS / lab06 / Landmark_Recognition_SIFT](https://github.com/asad-mashood/ComputerVision-LABS/tree/main/lab06/Landmark_Recognition_SIFT)**

# Lab 06: Landmark Recognition Using SIFT Algorithm

[![Student ID](https://img.shields.io/badge/Student%20ID-FA23--BAI--039-blue)](https://github.com/asad-mashood)
[![Student](https://img.shields.io/badge/Student-Muhammad%20Asad%20Mashood-darkblue)](https://github.com/asad-mashood)
[![Repository](https://img.shields.io/badge/Repository-ComputerVision--LABS-teal)](https://github.com/asad-mashood/ComputerVision-LABS)
[![Course](https://img.shields.io/badge/Course-Computer%20Vision-blueviolet)](https://github.com/asad-mashood/ComputerVision-LABS)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?logo=opencv)](https://opencv.org/)

**Student Name:** Muhammad Asad Mashood  
**Student ID:** FA23-BAI-039  
**Course:** Computer Vision  
**Parent Repository:** [ComputerVision-LABS](https://github.com/asad-mashood/ComputerVision-LABS)  
**Module:** `lab06/Landmark_Recognition_SIFT`  

---

## 1. Overview & Architecture

An automated landmark recognition and geometric localization application designed for tourism information centers:
- **SIFT Feature Extraction:** DoG scale-space extrema and 128-D orientation histograms.
- **Brute-Force Matching with Lowe's Ratio Test ($\tau = 0.75$):** Robust filtering of ambiguous feature matches.
- **RANSAC Homography & Geometric Sanity Checks:** Verifying convexity, positive determinant, and valid area ratios.
- **Rejection Gate ($T_{\text{match}} \ge 20$):** Automated rejection of non-database landmarks.

<div align="center">
  <img src="figures/pipeline_architecture.png" alt="Architecture Diagram" width="100%" />
</div>

---

## 2. Experimental Benchmark Results

| Condition | Ground Truth | Predicted Landmark | Query KPs | Matches | Inliers | Decision | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Landmark Image** | `eiffel_tower` | `eiffel_tower` | `2,946` | `2,946` | `2,946` | **RECOGNIZED** | `1368.0 ms` |
| **Rotated Landmark Image (45 deg)** | `eiffel_tower` | `eiffel_tower` | `2,517` | `1,234` | `1,227` | **RECOGNIZED** | `1131.8 ms` |
| **Resized Landmark Image (0.5x Scale)** | `eiffel_tower` | `eiffel_tower` | `672` | `509` | `508` | **RECOGNIZED** | `372.5 ms` |
| **Reduced Brightness Image (0.35x)** | `eiffel_tower` | `eiffel_tower` | `760` | `718` | `717` | **RECOGNIZED** | `392.5 ms` |
| **Different Viewpoint Image** | `eiffel_tower` | `eiffel_tower` | `1,020` | `1,020` | `1,020` | **RECOGNIZED** | `431.6 ms` |
| **Non-Database Landmark (Big Ben)** | `unknown` | `Unknown / Rejected` | `1,984` | `22` | `0` | **REJECTED** | `895.3 ms` |

---

## 3. Visual Results Gallery

### Task 1: Landmark Reference Database
Multi-angle reference gallery containing the 5 target landmarks:
![Reference Database](figures/task1_reference_database.png)

### Task 2: SIFT Keypoints & Orientation Field
Difference-of-Gaussians (DoG) keypoints rendered with scale circles and gradient orientation rays:
![SIFT Keypoints](figures/task2_sift_features.png)

### Task 4: Feature Matching & Non-Database Rejection
Side-by-side matching pairs comparing genuine match under rotation vs rejection of an unknown landmark (Big Ben):
![Feature Matching Comparison](figures/task4_feature_matching_comparison.png)

### Task 5 & 6: Environmental Robustness & Geometric Localization Matrix
Comprehensive 6-condition test matrix showing localized bounding polygons and rejection banners:
![Environmental Evaluation](figures/task6_environmental_evaluation.png)

### Benchmark Telemetry
![Benchmark Telemetry](figures/task6_benchmark_telemetry.png)

---

## 4. Execution Instructions

```bash
# Clone the repository
git clone https://github.com/asad-mashood/ComputerVision-LABS.git

# Navigate to lab06 directory
cd ComputerVision-LABS/lab06/Landmark_Recognition_SIFT

# Install dependencies
pip install -r requirements.txt

# Run the complete experimental pipeline
python run_lab06.py
```

---

## 5. Interactive Streamlit Web Application

An interactive web studio is included for real-time landmark recognition, pose localization, and telemetry visualization.

```bash
# Launch interactive Streamlit studio
streamlit run app.py
```

### Features:
- 📤 **Live Upload:** Drag-and-drop any tourist capture (JPG/PNG).
- 🧪 **Pre-configured Test Suite:** Instant benchmarking against Rotation, Resizing, Dark Illumination, and Non-Database Impostors.
- 🎛️ **Hyper-parameter Tuning:** Real-time sliders for Lowe's ratio test ($\tau$), Rejection Gate matches ($T_{\text{match}}$), and RANSAC inlier threshold.
- 🎯 **5-Tab Multi-Angle Analysis:** Geometric Localization, Feature Matches, SIFT Extrema, Candidate Database Ranking, and Database Reference Gallery.
