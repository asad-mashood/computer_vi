# Lab 05: HOG-Based Industrial Defect Classification

## Overview
This directory contains the experiments, results, and deployment code for Lab 05. The lab explores classical computer vision techniques for industrial steel surface inspection, specifically relying on **Histogram of Oriented Gradients (HOG)** features alongside machine learning classifiers (Linear SVM, RBF SVM, and Logistic SGD). 

The goal is to automatically classify steel surfaces into different defect categories (e.g., crazing, inclusion, patches, pitted surface, rolled-in scale, scratches).

## Directory Structure

| File/Directory | Description |
| :--- | :--- |
| `CV_Lab05_FA23_BAI_039_MuhammadAsadMashood.ipynb` | Main Jupyter Notebook containing step-by-step exploration, training, and testing of HOG feature extraction and models. |
| `Lab05_train.py` | Training script to reproduce results, evaluate hyperparameters, generate visual plots, and compile the final PDF report. |
| `app.py` | A Streamlit web application providing a live interface for steel defect inspection using a webcam or photo uploads. |
| `inspection_model.joblib` | The serialized trained machine learning model exported for use by `app.py`. |
| `Lab05_requirements.txt` | A list of Python package dependencies for the lab. |
| `lab05_results/` | Directory populated with evaluation CSVs, generated plots, and the detailed PDF report (`Lab05_Report.pdf`). |
| `Lab05_Results.zip` | The compressed archive of all experimental results. |

## Quick Start

### 1. Install Dependencies
Make sure your environment is activated and install the requirements:
```bash
pip install -r Lab05_requirements.txt
```
*Note: The NEU Steel Surface defect dataset will automatically download via `kagglehub` the first time you run the training script or notebook.*

### 2. Live Web App Inspection
You can test the trained model interactively using the Streamlit app. It supports both image upload and live camera classification:
```bash
streamlit run app.py
```

### 3. Run the Training Pipeline (Optional)
If you wish to re-train the models and regenerate the plots:
```bash
python Lab05_train.py
```

## Highlighted Results

Here are a few quick visual highlights generated during the experiment. Check out the `lab05_results/` folder for the full suite of metrics and the comprehensive `Lab05_Report.pdf`.

### HOG Feature Extraction
The HOG descriptor extracts robust edge information used for the classification.
![HOG Examples](./lab05_results/hog_examples.png)

### Model Comparison
Performance of top classifiers across different settings.
![Model Comparison](./lab05_results/model_comparison.png)

### Confusion Matrix
Detailed analysis of the chosen model's predictions over the held-out validation dataset.
![Confusion Matrix](./lab05_results/confusion_matrix_normalized.png)

### Robustness
Performance under simulated changes in imaging conditions (e.g., blur, noise, brightness adjustments).
![Robustness](./lab05_results/robustness_change.png)

---
*Developed by Muhammad Asad Mashood (FA23-BAI-039) for Computer Vision*
