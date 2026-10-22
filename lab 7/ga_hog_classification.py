"""
Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm
Author: Muhammad Asad Mashood
Dataset: DermaMNIST
"""

import os
import sys
import json
import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef
)

RANDOM_SEED = 99
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def compute_multiclass_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))

def main():
    print("================================================================")
    print("  LAB 07: HOG + GENETIC ALGORITHM MEDICAL CLASSIFICATION")
    print("  Author: Muhammad Asad Mashood")
    print("  Dataset: DermaMNIST")
    print("================================================================")
    
    results_path = os.path.join(os.path.dirname(__file__), "results", "metrics_comparison.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
            
        print("\n[1] Table 1: Configuration")
        print("  - Medical Dataset: DermaMNIST")
        print("  - Resolution: 128x128 | Orientations: 9 | Cell: 8x8 | Block: 2x2 | Normalization: L2-Hys")
        print("  - Total Original HOG Features:", data["n_total_features"])
        print("  - GA Selected Features:", data["n_selected_features"], f"({data['reduction_pct']}% reduction)")
        
        print("\n[2] Table 2: Main Performance Comparison")
        b = data["baseline_metrics"]
        p = data["proposed_metrics"]
        imp = data["improvements"]
        print(f"  {'Metric':<25} | {'HOG + SVM (Base)':<18} | {'GA-HOG + SVM (Prop)':<19} | {'Improvement':<12}")
        print("  " + "-"*82)
        print(f"  {'Accuracy (%)':<25} | {b['accuracy']:<18} | {p['accuracy']:<19} | {'+' if imp['accuracy']>0 else ''}{imp['accuracy']}%")
        print(f"  {'Precision (Macro, %)':<25} | {b['precision_macro']:<18} | {p['precision_macro']:<19} | {'+' if imp['precision_macro']>0 else ''}{imp['precision_macro']}%")
        print(f"  {'Recall (Macro, %)':<25} | {b['recall_macro']:<18} | {p['recall_macro']:<19} | {'+' if imp['recall_macro']>0 else ''}{imp['recall_macro']}%")
        print(f"  {'Macro-F1 (%)':<25} | {b['macro_f1']:<18} | {p['macro_f1']:<19} | {'+' if imp['macro_f1']>0 else ''}{imp['macro_f1']}%")
        print(f"  {'Weighted-F1 (%)':<25} | {b['weighted_f1']:<18} | {p['weighted_f1']:<19} | {'+' if imp['weighted_f1']>0 else ''}{imp['weighted_f1']}%")
        print(f"  {'Balanced Accuracy (%)':<25} | {b['balanced_accuracy']:<18} | {p['balanced_accuracy']:<19} | {'+' if imp['balanced_accuracy']>0 else ''}{imp['balanced_accuracy']}%")
        print(f"  {'Specificity (Macro, %)':<25} | {b['specificity_macro']:<18} | {p['specificity_macro']:<19} | {'+' if imp['specificity_macro']>0 else ''}{imp['specificity_macro']}%")
        print(f"  {'Cohen\'s Kappa':<25} | {b['cohen_kappa']:<18} | {p['cohen_kappa']:<19} | {'+' if imp['cohen_kappa']>0 else ''}{imp['cohen_kappa']}")
        print(f"  {'Matthews Corr Coeff':<25} | {b['matthews_corrcoef']:<18} | {p['matthews_corrcoef']:<19} | {'+' if imp['matthews_corrcoef']>0 else ''}{imp['matthews_corrcoef']}")
        print(f"  {'Inference Latency (ms)':<25} | {b['latency_ms']:<18} | {p['latency_ms']:<19} | {imp['latency_speedup_x']}x faster")
        print("  " + "-"*82)
        print("\nAll plots and figures are saved under the `figures/` directory.")

if __name__ == "__main__":
    main()
