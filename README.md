# 👁️ Computer Vision Portfolio: From Classical Optics to Deep Representation

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/asad-mashood/computer_vi/master/lab%206/app.py)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Student ID](https://img.shields.io/badge/Student%20ID-FA23--BAI--039-blue)](https://github.com/asad-mashood)

**Author:** Muhammad Asad Mashood  
**Course:** Computer Vision  
**Institution:** COMSATS University Islamabad  

A comprehensive, production-grade Computer Vision portfolio spanning classical spatial filtering, frequency-domain transformations, scale-space invariant feature matching, boundary optimization, industrial anomaly detection, and deep transfer learning.

---

## 🌟 Flagship Project: Lab 06 LandmarkAI Studio

> **[👉 Launch Live Interactive Studio on Streamlit Cloud](https://share.streamlit.io/asad-mashood/computer_vi/master/lab%206/app.py)**

<div align="center">
  <img src="lab%206/figures/pipeline_architecture.png" alt="Landmark Recognition Pipeline Architecture" width="100%" />
</div>

An automated landmark recognition and geometric localization application engineered for smart tourism and robotic vision:
- **SIFT Invariant Feature Engine:** Extracts Difference-of-Gaussians (DoG) scale-space extrema and 128-D orientation histograms.
- **Lowe's Ratio Matching ($\tau = 0.75$):** Ambiguity-free nearest-neighbor matching using Euclidean distance.
- **RANSAC Homography & Geometric Sanity Checks:** Validates projective matrix determinant sign, convexity of the mapped quadrilateral, and bounding area ratios.
- **Automated Distractor Rejection ($T_{\text{inliers}} \ge 20$):** High-confidence rejection of non-database landmarks and unknown distractors.

```bash
# Launch the flagship studio locally in 2 commands:
cd "lab 6"
streamlit run app.py
```

---

## 📚 Portfolio Curriculum & Laboratory Modules

| Module | Project Focus | Core Techniques | Key Artifacts | Status |
| :--- | :--- | :--- | :--- | :---: |
| [**Lab 01**](Lab1/) | **Image Representation & Transfer Learning** | Intensity Transformations, Contrast Stretching, Histogram Equalization | `Jupyter Notebook`, `Benchmark CSVs` | Completed |
| [**Lab 02**](Lab2/) | **Skin Lesion Segmentation & Filtering** | Gaussian Smoothing, Median Denoising, HAM10000 Macro-F1 Analysis | `Evaluation Curves`, `Confusion Matrices` | Completed |
| [**Lab 03**](lab3/) | **Edge Detection & CNN Cross-Representation** | Sobel, Laplacian, Canny, Gaussian Noise Stress-Testing | `15 Experiments CSV`, `Comparative Edge Maps` | Completed |
| [**Lab 04**](lab%204/) | **Boundary Optimization & Canny Thresholding** | 8-Method Comparative Extraction, Morphological Contours | `Stage Visualizations`, `Benchmark CSVs` | Completed |
| [**Lab 05**](Lab5/) | **Industrial Anomaly & Defect Inspection** | Statistical Texture Analysis, Logistic Regression, Streamlit App | `inspection_model.joblib`, `Interactive UI` | Completed |
| [**Lab 06**](lab%206/) | **SIFT Landmark Recognition & Homography Studio** | Scale-Space DoG, Lowe's Ratio Test, RANSAC Homography, Streamlit Studio | `Streamlit App`, `Docker`, `Telemetry Dashboard` | 🌟 **Flagship** |
| [**Lab 07**](lab%207/) | **GA Feature Selection for HOG Medical Classification** | Genetic Algorithms, HOG Optimization, DermaMNIST, SVM Classification | `Evolution Curves`, `Radar Analytics` | Completed |
| [**LAB_MID**](LAB_MID/) | **HOG vs SIFT vs ResNet-50 Blood Cell Classification** | Classical vs Deep Vision, BloodMNIST, In-Plane Rotation, Noise Bootstrap, Occlusion Maps | `Interactive Dashboard`, `Visual Gallery` | Completed |

---

## ⚡ Quickstart & Local Installation

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Portfolio
```bash
git clone https://github.com/asad-mashood/computer_vi.git
cd computer_vi
```

### 2. Run the Flagship Interactive Studio (Lab 06)
```bash
cd "lab 6"
pip install -r requirements.txt
streamlit run app.py
```

### 3. Run with Docker
```bash
cd "lab 6"
docker build -t landmark-sift .
docker run -p 8501:8501 landmark-sift
# Access at http://localhost:8501
```

---

## 📂 Repository Hierarchy

```
computer_vi/
├── Lab1/                         # Lab 1: Image Processing & Transfer Learning
├── Lab2/                         # Lab 2: Skin Lesion Segmentation & Filtering
├── lab3/                         # Lab 3: Edge Detection & Representation
├── lab 4/                        # Lab 4: Boundary Extraction & Canny Optimization
├── Lab5/                         # Lab 5: Real-Time Defect Inspection System
├── lab 6/                        # Lab 6: SIFT Landmark Recognition Studio (Flagship)
│   ├── .streamlit/               # Streamlit styling & performance configuration
│   ├── data/landmarks/           # Multi-angle reference database & test imagery
│   ├── figures/                  # Experimental visualizations & telemetry charts
│   ├── results/                  # Benchmarks in CSV and JSON formats
│   ├── app.py                    # Interactive Streamlit Web Studio
│   ├── Dockerfile                # Production Docker container setup
│   ├── LAB06_REPORT.md           # Formal academic report
│   ├── Lab06_SIFT_Landmark_Recognition.ipynb # Research Notebook
│   ├── README.md                 # Lab 6 comprehensive documentation
│   ├── requirements.txt          # Production dependencies
│   └── run_lab06.py              # CLI benchmark suite
├── lab 7/                        # Lab 7: GA-Optimized HOG Medical Image Classification
├── LAB_MID/                      # Lab Mid Exam: HOG vs SIFT vs ResNet-50 on BloodMNIST
│   ├── figures/                  # 28 High-resolution benchmark figures
│   ├── tables/                   # Raw CSV benchmark telemetry
│   ├── dashboard.html            # Interactive Browser Diagnostics Dashboard
│   ├── FA23_BAI_039_Muhammad_Asad_Mashood_LabMid(cv).ipynb # Primary executed notebook
│   └── README.md                 # Comprehensive midterm comparative study
└── README.md                     # Central portfolio showcase
```

---

## 📄 License & Attribution

This project is open-source under the [MIT License](LICENSE).  
Developed by **Muhammad Asad Mashood** ([@asad-mashood](https://github.com/asad-mashood)) for the Computer Vision laboratory curriculum.
