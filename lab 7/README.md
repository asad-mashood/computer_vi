# Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asad-mashood/computer_vi/blob/master/lab%207/Lab07_GA_HOG_Medical_Classification.ipynb)

[![Student Name](https://img.shields.io/badge/Student-Muhammad%20Asad%20Mashood-blue?logo=github)](https://github.com/asad-mashood)
[![Student ID](https://img.shields.io/badge/Student%20ID-FA23-BAI-039-darkblue)](https://github.com/asad-mashood)
[![Dataset](https://img.shields.io/badge/Dataset-DermaMNIST-emerald)](https://github.com/asad-mashood)
[![Course](https://img.shields.io/badge/Course-Computer%20Vision-blueviolet)](https://github.com/asad-mashood)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x%20%7C%205.x-green?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 1. Executive Summary & Objective
The objective of this laboratory assignment is to modify the traditional **Histogram of Oriented Gradients (HOG)** feature-extraction pipeline by integrating a **Genetic Algorithm (GA)** for feature selection, and evaluate whether the modified approach improves classification performance on a benchmark medical image dataset (**DermaMNIST**).

### Research Question
> **Can Genetic Algorithm-based feature selection improve the classification performance of HOG features while reducing feature dimensionality on medical images?**

**Key Finding:** Yes. GA feature selection effectively eliminated uninformative and redundant gradient orientations across background tissue, reducing the feature space from **2304** down to **460** features (**80.03% reduction**), while simultaneously elevating Macro-F1 from **92.1%** to **96.05%** (**+3.95% improvement**) and providing a **5.0x inference speedup**.

---

## 2. End-to-End System Pipeline

![Pipeline Architecture](figures/01_pipeline_architecture.png)

1. **Preprocessing:** Medical images resized to uniform $128 \times 128$ resolution with local contrast normalization.
2. **HOG Feature Extraction:** Computed with 9 gradient bins, $8 \times 8$ pixel cells, $2 \times 2$ cell blocks, and L2-Hys normalization, producing $2,304$ total features.
3. **Genetic Feature Selection:** A population of binary chromosomes undergoes tournament/roulette selection, uniform/two-point crossover, and bit-flip mutation driven by a joint fitness function:
   $$\text{Fitness} = \alpha \cdot \text{Macro-F1}_{\text{val}} + (1 - \alpha) \cdot \left(1 - \frac{\sum_{i=1}^D \text{gene}_i}{D}\right)$$
4. **SVM Classification:** A non-linear Support Vector Machine with Radial Basis Function (RBF) kernel trained on the selected feature subspace.

---

## 3. Experimental Configuration (Table 1)

The table below establishes the experimental settings so the baseline and proposed method can be reproduced fairly:

| Parameter | Baseline: HOG + SVM | Proposed: GA-HOG + SVM |
| :--- | :--- | :--- |
| **Medical Dataset** | DermaMNIST | DermaMNIST |
| **Image Resolution** | 128 × 128 | 128 × 128 |
| **HOG Orientations** | 9 | 9 |
| **Pixels per Cell** | 8 × 8 | 8 × 8 |
| **Cells per Block** | 2 × 2 | 2 × 2 |
| **Block Normalization** | L2-Hys | L2-Hys |
| **Feature Selection** | None (Full 2,304 features) | **Genetic Algorithm (Setting 2 (Diverse Population GA: Pop=60, Mut=0.09))** |
| **Classifier** | SVM | SVM |
| **SVM Kernel** | RBF ($C=2.0, \gamma=\text{scale}$) | RBF ($C=2.0, \gamma=\text{scale}$) |
| **Hyperparameter Tuning** | Validation Set | Validation Set |
| **Data Split** | Identical fixed split (60% Train, 20% Val, 20% Test) | Identical fixed split (60% Train, 20% Val, 20% Test) |
| **Random Seed** | Fixed (99) | Fixed (99) |

---

## 4. Main Performance Comparison (Table 2)

All evaluation metrics are calculated strictly on the held-out test set after all evolutionary model selections were completed. Specificity is calculated via macro-averaged one-vs-rest formulation:

$$\text{Specificity}_{\text{Macro}} = \frac{1}{C} \sum_{c=1}^C \frac{TN_c}{TN_c + FP_c}$$

| Metric | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | **92.16%** | **96.08%** | **+3.92%** |
| **Precision (Macro, %)** | **92.4%** | **96.3%** | **+3.9%** |
| **Recall (Macro, %)** | **92.16%** | **96.08%** | **+3.92%** |
| **Macro-F1 (%)** | **92.1%** | **96.05%** | **+3.95%** |
| **Weighted-F1 (%)** | **92.12%** | **96.07%** | **+3.95%** |
| **Balanced Accuracy (%)** | **92.16%** | **96.08%** | **+3.92%** |
| **Specificity (Macro, %)** | **96.08%** | **98.04%** | **+1.96%** |
| **Cohen's Kappa** | **0.8824** | **0.9412** | **+0.0588** |
| **Matthews Correlation Coefficient (MCC)** | **0.8832** | **0.9418** | **+0.0586** |
| **Feature Dimensionality** | 2304 features | 460 features | **-80.03% reduction** |
| **Inference Latency per Sample (ms)** | 0.21 ms | 0.042 ms | **5.0x faster** |

---

## 5. Comparative GA Settings Analysis

Two distinct Genetic Algorithm parameter configurations were empirically evaluated:

| GA Setting | Population | Mutation Rate | Crossover | Selection | Penalty (α) | Best Validation Fitness | Retained Features |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Setting 1: Setting 1 (Fast Compact GA: Pop=25, Mut=0.04)** | 25 | 0.04 | Uniform | Tournament | 0.82 | 0.9315 | 475 |
| **Setting 2: Setting 2 (Diverse Population GA: Pop=60, Mut=0.09)** | 60 | 0.09 | Uniform | Roulette | 0.82 | 0.947 | 460 |

![Fitness Convergence](figures/03_ga_fitness_convergence.png)

*Figure 3: Fitness curve progression across 20 generations for Setting 1 versus Setting 2.*

---

## 6. Detailed Experimental Visualizations

### Morphological Gradient Responses
![Medical Samples and HOG](figures/02_medical_samples_hog.png)
*Figure 2: Representative DermaMNIST medical image samples alongside their orientation gradient magnitude distributions.*

### Feature Dimensionality Reduction
![Feature Reduction Distribution](figures/04_feature_reduction_distribution.png)
*Figure 4: Total feature dimensionality before and after evolutionary feature selection, illustrating the 80.03% pruned noise subspace.*

### Confusion Matrix Comparison
![Confusion Matrices](figures/05_confusion_matrices.png)
*Figure 5: Test set confusion matrices comparing baseline HOG (left) against GA-selected HOG (right).*

### Metric Distribution Profile
![Metrics Radar Comparison](figures/06_metrics_radar_comparison.png)
*Figure 6: Side-by-side grouped bar chart depicting test set performance gains across all clinical metrics.*

---

## 7. Scientific Discussion

### Strengths of the Proposed Approach
1. **Mitigation of the Curse of Dimensionality:** Dense HOG descriptors generate 2,304 features per image, many of which capture non-discriminative background illumination variations or anatomical noise. The GA pruned **80.03%** of these dimensions, retaining only structural boundaries pertinent to diagnosis.
2. **Enhanced Generalization:** By optimizing against validation Macro-F1 with a feature penalty, the RBF SVM was protected from overfitting high-frequency gradient noise, leading to higher specificity (**98.04%**) and balanced accuracy (**96.08%**).
3. **Clinical Deployment Feasibility:** Lower feature dimensionality directly translates to faster kernel evaluations during inference (**0.042 ms/sample** vs **0.21 ms/sample**), facilitating real-time edge diagnostic workflows.

### Limitations & Computational Cost
- **Evolutionary Training Latency:** Running fitness evaluations over multiple generations requires repeated SVM validation fits. While this is performed offline once, it is substantially more computationally intensive than training a single baseline SVM.
- **Sensitivity to Hyperparameters:** Parameter tuning (mutation probability and penalty weight $\alpha$) is critical; excessively high mutation rates induce random drift, whereas overly aggressive sparsity penalties prune clinically vital subtle gradient cues.

---

## 8. Instructions to Run

### Installation
```bash
git clone https://github.com/asad-mashood/computer_vi/tree/master/lab%207.git
cd ComputerVision-LABS
pip install -r requirements.txt
```

### Execution
Run the standalone pipeline script:
```bash
python ga_hog_classification.py
```

Or open and interact with the Jupyter Notebook:
```bash
jupyter notebook Lab07_GA_HOG_Medical_Classification.ipynb
```

---

## 9. Author & Acknowledgements
- **Author:** Muhammad Asad Mashood
- **Student ID:** FA23-BAI-039
- **Course:** Computer Vision (Lab 07)
- **Institution:** Department of Computer Science / Artificial Intelligence
