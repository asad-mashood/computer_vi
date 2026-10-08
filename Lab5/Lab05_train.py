"""Lab 05 training, evaluation, graphs, report, and Streamlit model export.

Put this file beside app.py and run with: py Lab05_train.py
Install packages first: py -m pip install -r Lab05_requirements.txt
The public dataset downloads automatically on the first run.
"""

from pathlib import Path
from collections import Counter
import hashlib
import re
import shutil

import kagglehub
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage import exposure, filters, transform, util
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
                               Spacer, PageBreak, Image as ReportImage)

SEED = 42
IMAGE_SIZE = (80, 80)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'lab05_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# Leave as None to download the public dataset. If it is already extracted,
# set this to its containing folder, for example Path('/content/NEU-DET').
DATASET_ROOT = None

# Optional: a folder containing real defect-free steel images.
# This is required to train and evaluate a genuine ACCEPT decision.
NORMAL_IMAGES_DIR = BASE_DIR / 'normal_images'

if DATASET_ROOT is None:
    DATASET_ROOT = Path(kagglehub.dataset_download(
        'sovitrath/neu-steel-surface-defect-detect-trainvalid-split'
    ))
else:
    DATASET_ROOT = Path(DATASET_ROOT).expanduser()
print('Dataset location:', DATASET_ROOT)

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
CLASS_ALIASES = {
    'crazing': 'crazing', 'inclusion': 'inclusion', 'patches': 'patches',
    'pitted_surface': 'pitted surface', 'pitted': 'pitted surface',
    'rolled_in_scale': 'rolled-in scale', 'rolledin_scale': 'rolled-in scale',
    'scratches': 'scratches', 'scratch': 'scratches',
    'normal': 'normal', 'non_defective': 'normal', 'defect_free': 'normal',
}

def find_split(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Dataset folder does not exist: {root}')
    for parent in [root, *root.rglob('*')]:
        if not parent.is_dir():
            continue
        for train_name, valid_name in [('train_images', 'valid_images'),
                                       ('train/images', 'valid/images'),
                                       ('train/images', 'val/images'),
                                       ('train/images', 'validation/images')]:
            train_dir, valid_dir = parent / train_name, parent / valid_name
            if train_dir.is_dir() and valid_dir.is_dir():
                return train_dir, valid_dir
    raise FileNotFoundError(
        'Could not find train_images and valid_images under the downloaded folder. '
        'Print the directory structure and set DATASET_ROOT to the extracted dataset root.'
    )

def label_from_path(path, split_dir):
    # Class folders are supported; the linked dataset normally uses filename prefixes.
    if path.parent != split_dir:
        folder = re.sub(r'[^a-z0-9]+', '_', path.parent.name.lower()).strip('_')
        if folder in CLASS_ALIASES:
            return CLASS_ALIASES[folder]
    stem = re.sub(r'[_-]\d+.*$', '', path.stem.lower())
    key = re.sub(r'[^a-z0-9]+', '_', stem).strip('_')
    if key not in CLASS_ALIASES:
        raise ValueError(f'Unrecognized defect label in filename: {path.name}')
    return CLASS_ALIASES[key]

def image_files(folder):
    return sorted(p for p in Path(folder).rglob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)

train_dir, valid_dir = find_split(DATASET_ROOT)
train_paths, valid_paths = image_files(train_dir), image_files(valid_dir)
if not train_paths or not valid_paths:
    raise ValueError('The training or validation image directory is empty.')

y_train = [label_from_path(p, train_dir) for p in train_paths]
y_valid = [label_from_path(p, valid_dir) for p in valid_paths]

if NORMAL_IMAGES_DIR is not None:
    normal_paths = image_files(NORMAL_IMAGES_DIR)
    if len(normal_paths) < 10:
        raise ValueError('Provide at least 10 real normal images for a reliable split.')
    normal_train, normal_valid = train_test_split(
        normal_paths, test_size=0.2, random_state=SEED
    )
    train_paths += list(normal_train)
    valid_paths += list(normal_valid)
    y_train += ['normal'] * len(normal_train)
    y_valid += ['normal'] * len(normal_valid)

train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in train_paths}
duplicated = [p.name for p in valid_paths
              if hashlib.sha256(p.read_bytes()).hexdigest() in train_hashes]
if duplicated:
    raise ValueError(f'Duplicate image bytes across train/validation: {duplicated[:5]}')

classes = sorted(set(y_train))
if set(y_valid) != set(y_train):
    raise ValueError('Training and validation must contain the same classes.')
if min(Counter(y_train).values()) < 3:
    raise ValueError('Every training class needs at least three examples.')

counts = pd.DataFrame({
    'train': pd.Series(Counter(y_train)),
    'validation': pd.Series(Counter(y_valid)),
}).fillna(0).astype(int).sort_index()
print('Classes:', ', '.join(classes))
print(counts.to_string())
counts.to_csv(OUTPUT_DIR / 'dataset_counts.csv')

def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L').resize(IMAGE_SIZE, Image.Resampling.BILINEAR))

train_images = np.stack([read_gray(p) for p in train_paths])
valid_images = np.stack([read_gray(p) for p in valid_paths])
print('Resized grayscale arrays:', train_images.shape, valid_images.shape)

fig, axes = plt.subplots(1, len(classes), figsize=(2.3 * len(classes), 2.8))
for ax, label in zip(np.atleast_1d(axes), classes):
    i = y_train.index(label)
    ax.imshow(train_images[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(label, fontsize=9)
    ax.axis('off')
fig.suptitle('Representative training images')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'sample_images.png', dpi=160, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
positions = np.arange(len(counts))
ax.bar(positions - 0.18, counts['train'], width=0.36, label='Train', color='#167d9d')
ax.bar(positions + 0.18, counts['validation'], width=0.36,
       label='Held-out validation', color='#e9a43b')
ax.set_xticks(positions, counts.index, rotation=25, ha='right')
ax.set(ylabel='Image count', title='Number of images in each defect category')
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=170)
plt.close(fig)

def hog_features(images, cell_size, orientations):
    rows = [hog(image, orientations=orientations,
                pixels_per_cell=(cell_size, cell_size),
                cells_per_block=(2, 2), block_norm='L2-Hys',
                transform_sqrt=True, feature_vector=True)
            for image in images]
    return np.asarray(rows, dtype=np.float32)

first_by_class = [y_train.index(label) for label in classes]
fig, axes = plt.subplots(len(classes), 2, figsize=(7, 2.35 * len(classes)))
for row, idx in enumerate(first_by_class):
    _, rendered = hog(train_images[idx], orientations=9,
                      pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      block_norm='L2-Hys', transform_sqrt=True,
                      visualize=True)
    axes[row, 0].imshow(train_images[idx], cmap='gray')
    axes[row, 1].imshow(exposure.rescale_intensity(rendered), cmap='gray')
    axes[row, 0].set_ylabel(y_train[idx])
    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])
axes[0, 0].set_title('Resized grayscale')
axes[0, 1].set_title('HOG edges')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'hog_examples.png', dpi=160, bbox_inches='tight')
plt.close(fig)

indices = np.arange(len(y_train))
fit_idx, dev_idx = train_test_split(
    indices, test_size=0.2, stratify=y_train, random_state=SEED
)
y_array = np.asarray(y_train)

def make_classifier(name):
    if name == 'Linear SVM':
        estimator = LinearSVC(C=0.5, class_weight='balanced',
                              max_iter=5000, random_state=SEED)
    elif name.startswith('RBF SVM C='):
        estimator = SVC(kernel='rbf', C=float(name.rsplit('=', 1)[1]),
                        gamma='scale', class_weight='balanced')
    elif name == 'Logistic SGD':
        estimator = SGDClassifier(loss='log_loss', alpha=0.0001,
                                  class_weight='balanced', max_iter=1500,
                                  tol=1e-3, random_state=SEED)
    else:
        raise ValueError(f'Unknown model: {name}')
    return make_pipeline(StandardScaler(with_mean=False), estimator)

sweep_rows = []
# Fast first pass: all nine HOG combinations for both linear classifiers.
for cell_size in (4, 8, 16):
    for orientations in (6, 9, 12):
        features = hog_features(train_images, cell_size, orientations)
        for model_name in ('Linear SVM', 'Logistic SGD'):
            model = make_classifier(model_name)
            model.fit(features[fit_idx], y_array[fit_idx])
            predictions = model.predict(features[dev_idx])
            sweep_rows.append({
                'cell_size': cell_size, 'orientations': orientations,
                'model': model_name,
                'dev_accuracy': accuracy_score(y_array[dev_idx], predictions),
                'dev_macro_f1': f1_score(y_array[dev_idx], predictions,
                                          average='macro', zero_division=0),
                'feature_count': features.shape[1],
            })
        del features

shortlist = (pd.DataFrame(sweep_rows)
             .sort_values(['dev_macro_f1', 'dev_accuracy'], ascending=False)
             .drop_duplicates(['cell_size', 'orientations'])
             .head(3))
# Nonlinear SVMs are tried only on configurations chosen from training data.
for candidate in shortlist.itertuples():
    features = hog_features(train_images, candidate.cell_size, candidate.orientations)
    for model_name in ('RBF SVM C=1', 'RBF SVM C=3'):
        model = make_classifier(model_name)
        model.fit(features[fit_idx], y_array[fit_idx])
        predictions = model.predict(features[dev_idx])
        sweep_rows.append({
            'cell_size': candidate.cell_size, 'orientations': candidate.orientations,
            'model': model_name,
            'dev_accuracy': accuracy_score(y_array[dev_idx], predictions),
            'dev_macro_f1': f1_score(y_array[dev_idx], predictions,
                                      average='macro', zero_division=0),
            'feature_count': features.shape[1],
        })
    del features

sweep = pd.DataFrame(sweep_rows).sort_values(
    ['dev_macro_f1', 'dev_accuracy', 'feature_count'],
    ascending=[False, False, True]
).reset_index(drop=True)
sweep.to_csv(OUTPUT_DIR / 'hog_parameter_comparison.csv', index=False)
print('Development results (only this split chooses the model):')
print(sweep.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
chosen = sweep.iloc[0]
chosen_cell = int(chosen['cell_size'])
chosen_orientations = int(chosen['orientations'])
chosen_model_name = chosen['model']
print(f'\nSelected on development data: {chosen_model_name}, '
      f'{chosen_cell}×{chosen_cell} cells, {chosen_orientations} orientations')

# Heatmap: best measured development F1 for each HOG combination.
heat = (sweep.groupby(['cell_size', 'orientations'])['dev_macro_f1'].max()
        .unstack().reindex(index=[4, 8, 16], columns=[6, 9, 12]))
fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(heat.to_numpy(), cmap='YlGnBu', vmin=0, vmax=1)
ax.set_xticks(range(3), heat.columns)
ax.set_yticks(range(3), heat.index)
ax.set(xlabel='Orientation bins', ylabel='Pixels per cell',
       title='Best development macro F1 by HOG setting')
for row in range(3):
    for col in range(3):
        ax.text(col, row, f'{heat.iloc[row, col]:.3f}',
                ha='center', va='center', color='white' if heat.iloc[row, col] > 0.65 else 'black')
fig.colorbar(im, ax=ax, label='Macro F1')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'hog_heatmap.png', dpi=180)
plt.close(fig)

top = sweep.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
labels = [f'{r.model} | {r.cell_size}px / {r.orientations} bins'
          for r in top.itertuples()]
ax.barh(labels, top.dev_macro_f1, color='#257da5')
ax.set(xlim=(0, 1), xlabel='Development macro F1',
       title='Leading model and HOG combinations')
ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'hog_top_models.png', dpi=170)
plt.close(fig)

X_train = hog_features(train_images, chosen_cell, chosen_orientations)
X_valid = hog_features(valid_images, chosen_cell, chosen_orientations)

fitted_models = {}
final_rows = []
rbf_names = [name for name in ('RBF SVM C=1', 'RBF SVM C=3')
             if name in set(sweep['model'])]
best_rbf = (sweep[sweep['model'].isin(rbf_names)].iloc[0]['model']
            if rbf_names else None)
final_names = ['Linear SVM', 'Logistic SGD'] + ([best_rbf] if best_rbf else [])
for name in final_names:
    base = make_classifier(name)
    model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    model.fit(X_train, y_train)
    fitted_models[name] = model
    pred = model.predict(X_valid)
    final_rows.append({
        'model': name,
        'accuracy': accuracy_score(y_valid, pred),
        'macro_precision': precision_score(y_valid, pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_valid, pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_valid, pred, average='macro', zero_division=0),
    })

comparison = pd.DataFrame(final_rows)
comparison.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
print(comparison.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

chosen_model = fitted_models[chosen_model_name]
y_pred = chosen_model.predict(X_valid)
report_text = classification_report(y_valid, y_pred, labels=classes, zero_division=0)
print('\nSelected model classification report:\n', report_text)
(OUTPUT_DIR / 'classification_report.txt').write_text(report_text, encoding='utf-8')
per_class = pd.DataFrame(classification_report(
    y_valid, y_pred, labels=classes, output_dict=True, zero_division=0
)).T
per_class.to_csv(OUTPUT_DIR / 'per_class_metrics.csv')
probabilities = chosen_model.predict_proba(X_valid)
review = pd.DataFrame({
    'file': [p.name for p in valid_paths], 'true_class': y_valid,
    'predicted_class': y_pred, 'confidence': probabilities.max(axis=1),
})
review['correct'] = review['true_class'] == review['predicted_class']
review.to_csv(OUTPUT_DIR / 'validation_predictions.csv', index=False)
confusion_counts = pd.crosstab(
    pd.Categorical(y_valid, categories=classes),
    pd.Categorical(y_pred, categories=classes),
    rownames=['Actual'], colnames=['Predicted'], dropna=False,
)
confusion_counts.to_csv(OUTPUT_DIR / 'confusion_matrix.csv')
print('Example predictions (including any mistakes first):')
print(review.sort_values(['correct', 'confidence']).head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
palette = ['#136b86' if name == chosen_model_name else '#7aaac0'
           for name in comparison['model']]
bars = ax.bar(comparison['model'], comparison['macro_f1'], color=palette)
ax.bar_label(bars, fmt='%.3f', padding=4)
ax.set(ylim=(0, 1.08), ylabel='Held-out macro F1',
       title='Final comparison on the untouched validation split')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
category_f1 = per_class.loc[classes, 'f1-score']
ax.barh(category_f1.index, category_f1, color='#e2a545')
ax.set(xlim=(0, 1.05), xlabel='F1 score',
       title=f'Per-category validation F1 — {chosen_model_name}')
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'per_class_f1.png', dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_valid, y_pred, labels=classes,
                                         ax=ax, xticks_rotation=35,
                                         colorbar=False, cmap='Blues')
ax.set_title(f'Validation confusion matrix — {chosen_model_name}')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_valid, y_pred, labels=classes,
                                         normalize='true', values_format='.2f',
                                         ax=ax, xticks_rotation=35,
                                         colorbar=False, cmap='Blues')
ax.set_title(f'Normalized confusion matrix — {chosen_model_name}')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'confusion_matrix_normalized.png', dpi=170)
plt.close(fig)

rng = np.random.default_rng(SEED)

def perturb(images, condition):
    if condition == 'brightness':
        return np.clip(images.astype(np.float32) * 1.35 + 15, 0, 255).astype(np.uint8)
    if condition == 'gaussian_noise':
        return np.clip(images.astype(np.float32) +
                       rng.normal(0, 20, size=images.shape), 0, 255).astype(np.uint8)
    if condition == 'rotation_10_degrees':
        return np.stack([transform.rotate(im, 10, mode='reflect',
                                           preserve_range=True).astype(np.uint8)
                         for im in images])
    if condition == 'gaussian_blur':
        return np.stack([np.clip(filters.gaussian(im, sigma=1.2,
                                                  preserve_range=True), 0, 255).astype(np.uint8)
                         for im in images])
    raise ValueError(condition)

clean_accuracy = accuracy_score(y_valid, y_pred)
clean_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
robust_rows = [{'condition': 'clean', 'accuracy': clean_accuracy,
                'macro_f1': clean_f1, 'accuracy_change': 0.0, 'f1_change': 0.0}]
for condition in ('brightness', 'gaussian_noise', 'rotation_10_degrees', 'gaussian_blur'):
    altered = perturb(valid_images, condition)
    altered_features = hog_features(altered, chosen_cell, chosen_orientations)
    altered_pred = chosen_model.predict(altered_features)
    accuracy = accuracy_score(y_valid, altered_pred)
    macro_f1 = f1_score(y_valid, altered_pred, average='macro', zero_division=0)
    robust_rows.append({
        'condition': condition, 'accuracy': accuracy, 'macro_f1': macro_f1,
        'accuracy_change': accuracy - clean_accuracy,
        'f1_change': macro_f1 - clean_f1,
    })
robustness = pd.DataFrame(robust_rows)
robustness.to_csv(OUTPUT_DIR / 'robustness.csv', index=False)
print(robustness.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

fig, ax = plt.subplots(figsize=(9, 4))
positions = np.arange(len(robustness))
ax.bar(positions - 0.18, robustness.accuracy, 0.36, label='Accuracy')
ax.bar(positions + 0.18, robustness.macro_f1, 0.36, label='Macro F1')
ax.set_xticks(positions, robustness.condition, rotation=20, ha='right')
ax.set_ylim(0, 1)
ax.set_title('Validation performance under imaging changes')
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'robustness.png', dpi=160)
plt.close(fig)

changed = robustness.iloc[1:]
fig, ax = plt.subplots(figsize=(9, 4))
positions = np.arange(len(changed))
ax.bar(positions - 0.18, changed['accuracy_change'], 0.36,
       label='Accuracy change', color='#157a9b')
ax.bar(positions + 0.18, changed['f1_change'], 0.36,
       label='Macro F1 change', color='#e6a23c')
ax.axhline(0, color='#333333', linewidth=0.8)
ax.set_xticks(positions, changed.condition, rotation=15, ha='right')
ax.set(ylabel='Change from clean images (percentage points as fractions)',
       title='How each imaging change affects the selected model')
if np.allclose(changed[['accuracy_change', 'f1_change']].to_numpy(), 0):
    ax.text(0.5, 0.72, 'No change measured on these images',
            transform=ax.transAxes, ha='center', fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'robustness_change.png', dpi=170)
plt.close(fig)

def inspect_product(image_path):
    image = read_gray(Path(image_path))
    features = hog_features(image[None, ...], chosen_cell, chosen_orientations)
    label = str(chosen_model.predict(features)[0])
    probabilities = chosen_model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))
    action = 'ACCEPT PRODUCT' if label == 'normal' else 'REJECT PRODUCT'
    print('PRODUCT INSPECTION RESULT')
    print(f'Prediction: {label.upper()}')
    print(f'Estimated confidence: {confidence:.1%}')
    print(f'Action: {action}')
    if 'normal' not in classes:
        print('Note: This dataset has no normal examples; acceptance is not validated.')
    return {'prediction': label, 'estimated_confidence': confidence, 'action': action}

# Demonstrate on one held-out image without training on it.
inspection_example = inspect_product(valid_paths[0])
model_bundle = {
    'model': chosen_model, 'classes': classes, 'image_size': IMAGE_SIZE,
    'hog_cell_size': chosen_cell, 'hog_orientations': chosen_orientations,
    'model_name': chosen_model_name,
    'scikit_learn_version': __import__('sklearn').__version__,
    'scikit_image_version': __import__('skimage').__version__,
    'note': 'Use the same grayscale resizing and HOG settings for future images.',
}
joblib.dump(model_bundle, OUTPUT_DIR / 'inspection_model.joblib', compress=3)
shutil.copy2(OUTPUT_DIR / 'inspection_model.joblib', BASE_DIR / 'inspection_model.joblib')
print('Saved the trained model for later use.')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='LabTitle', parent=styles['Title'],
                          fontName='Helvetica-Bold', fontSize=16, spaceAfter=12))
styles.add(ParagraphStyle(name='LabBody', parent=styles['BodyText'],
                          fontSize=9, leading=13, spaceAfter=8))

def para(text):
    return Paragraph(text, styles['LabBody'])

def heading(text):
    return Paragraph(text, styles['Heading2'])

def report_table(frame, columns, digits=3):
    data = [columns]
    for row in frame.itertuples(index=False, name=None):
        data.append([f'{v:.{digits}f}' if isinstance(v, (float, np.floating))
                     else str(v) for v in row])
    table = Table(data, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#15365d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 7.3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#edf2f7')]),
        ('GRID', (0, 0), (-1, -1), 0.2, colors.HexColor('#b8c6d5')),
    ]))
    return table

pdf_path = OUTPUT_DIR / 'Lab05_Report.pdf'
document = SimpleDocTemplate(str(pdf_path), pagesize=(612, 792),
                             leftMargin=50, rightMargin=50,
                             topMargin=42, bottomMargin=42)
story = []
story += [Paragraph('Lab 05: HOG-Based Industrial Defect Classification', styles['LabTitle']),
          para('Muhammad Asad Mashood | FA23-BAI-039')]
story += [heading('Introduction and industrial application'),
          para('Manual steel surface inspection is slow and can vary across operators. '
               'This experiment assesses a HOG-based image classifier as an inspection aid. '
               'It predicts a primary defect category for an entire surface image.'),
          heading('Dataset description'),
          para(f'The NEU steel surface dataset provides {len(train_paths)} training and '
               f'{len(valid_paths)} held-out validation images across {len(classes)} labels: '
               f'{", ".join(classes)}. The image filename prefix supplies its primary '
               'defect label. The source has no normal images unless a separate folder '
               'was supplied explicitly.'),
          report_table(counts.reset_index(), ['Category', 'Train', 'Validation'], digits=0),
          Spacer(1, 10),
          ReportImage(str(OUTPUT_DIR / 'class_distribution.png'), width=450, height=200),
          heading('Preprocessing and HOG features'),
          para('Images are converted to grayscale and resized to 80 x 80 pixels. '
               'HOG summarizes local edge directions in normalized blocks of 2 x 2 cells. '
               'The experiment compares cell widths of 4, 8, and 16 pixels and 6, 9, '
               'and 12 orientation bins; square-root intensity transformation is enabled.')]

story += [PageBreak(), heading('Classification methodology and experimental setup'),
          para('Linear SVM, RBF SVM, and logistic-loss SGD classifiers receive HOG features. '
               'Twenty percent of '
               'the training data, stratified by class, is used for parameter selection. '
               'The untouched provided validation split is used once after selection. '
               'Scaling is fitted using only the corresponding training partition.'),
          para(f'The selected development setting was {chosen_model_name}, '
               f'{chosen_cell} x {chosen_cell} pixels per cell and '
               f'{chosen_orientations} orientations. Selection used macro F1; '
               'confidence is estimated by three-fold sigmoid calibration '
               'on the training data. RBF SVM was tried on the three HOG '
               'settings shortlisted by the linear classifiers.'),
          heading('HOG parameter results'),
          report_table(sweep[['cell_size', 'orientations', 'model', 'dev_macro_f1']].head(7),
                       ['Cell', 'Bins', 'Classifier', 'Dev macro F1']),
          Spacer(1, 10),
          ReportImage(str(OUTPUT_DIR / 'hog_heatmap.png'), width=355, height=250)]

story += [PageBreak(), heading('Held-out validation results'),
          para('The table compares all final classifiers using the chosen HOG setting. '
               'Precision, recall, and F1 are macro averages; accuracy is the '
               'fraction of correctly classified validation images.'),
          report_table(comparison, ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1']),
          Spacer(1, 8), heading(f'Confusion matrix: {chosen_model_name}'),
          ReportImage(str(OUTPUT_DIR / 'confusion_matrix.png'), width=290, height=254),
          ReportImage(str(OUTPUT_DIR / 'per_class_f1.png'), width=445, height=178),
          para('Rows are true image labels; columns are predicted image labels. '
               'Per-class precision, recall, F1 and support are also saved '
               'in per_class_metrics.csv and classification_report.txt.')]

worst = robustness.iloc[1:].sort_values('f1_change').iloc[0]
story += [PageBreak(), heading('Robustness analysis'),
          para('Brightness (+35% and +15 intensity), Gaussian noise (standard '
               'deviation 20), 10-degree rotation, and Gaussian blur (sigma 1.2) '
               'were applied separately to the held-out images. '
               'No retraining was done on the changed images.'),
          report_table(robustness, ['Condition', 'Accuracy', 'Macro F1', 'Acc change', 'F1 change']),
          Spacer(1, 10),
          ReportImage(str(OUTPUT_DIR / 'robustness_change.png'), width=455, height=206),
          para(f'The greatest observed macro F1 decrease in these tested conditions '
               f'was {abs(min(0, worst.f1_change)):.3f} for {worst.condition}. '
               'These are illustrative imaging changes; different factories or '
               'camera setups may have different error rates.')]

story += [PageBreak(), heading('Quality-control decision'),
          para(f'On a sample held-out image the prediction was '
               f'{inspection_example["prediction"]} with estimated confidence '
               f'{inspection_example["estimated_confidence"]:.1%}; the prototype action '
               f'was {inspection_example["action"]}. This is a classification result '
               'for one image, not a production acceptance guarantee.'),
          heading('Industrial deployment discussion'),
          para('An inspection station could preprocess each camera frame, compute HOG, '
               'predict the defect type, and display an estimated confidence. '
               'Before any automatic production action, collect representative '
               'normal images and choose a review threshold using independent '
               'factory data. Monitor camera drift and confirm decisions with operators.'),
          heading('Limitations'),
          para('The original dataset contains only defects, so it cannot establish '
               'accuracy for accepting good products. The supplied train/validation '
               'split may share production conditions, limiting generalization. '
               'Image-level classification '
               'does not identify defect locations and may miss multiple defect types '
               'in one image. Internal probability calibration is not a safety guarantee.'),
          heading('Conclusion'),
          para(f'The selected {chosen_model_name} achieved '
               f'{clean_accuracy:.3f} validation accuracy and {clean_f1:.3f} macro F1. '
               'The results support a prototype for defect-type identification; '
               'a measured accept/reject system needs normal samples and additional '
               'independent testing. The notebook exports its trained model, '
               'per-image predictions, result tables, graphs, and report.'),
          para('Dataset: https://www.kaggle.com/datasets/sovitrath/'
               'neu-steel-surface-defect-detect-trainvalid-split')]

def page_footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#506078'))
    canvas.drawString(50, 25, 'CV Lab 05 — industrial surface inspection')
    canvas.drawRightString(562, 25, f'Page {doc.page}')

document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
print('Saved report:', pdf_path.resolve())
print('Saved all results in:', OUTPUT_DIR.resolve())

# Copy the companion Streamlit app into the downloaded results archive.
app_path = BASE_DIR / 'app.py'
if not app_path.is_file():
    raise FileNotFoundError('Put app.py beside Lab05_train.py before running.')
shutil.copy2(app_path, OUTPUT_DIR / 'app.py')
requirements = ('streamlit\nstreamlit-webrtc\nav\n'
                f'scikit-image=={__import__("skimage").__version__}\n'
                f'scikit-learn=={__import__("sklearn").__version__}\n'
                'numpy\npandas\nPillow\njoblib\n')
(OUTPUT_DIR / 'requirements.txt').write_text(requirements, encoding='utf-8')
(OUTPUT_DIR / 'README.md').write_text(
    'Open a terminal in this folder, then run: py -m pip install -r requirements.txt\n'
    'Then run: py -m streamlit run app.py\n', encoding='utf-8')
print('Streamlit app and matching requirements added to results.')

archive_path = Path(shutil.make_archive(str(BASE_DIR / 'Lab05_Results'), 'zip',
                                        root_dir=BASE_DIR, base_dir=OUTPUT_DIR.name))
print('Results and report saved in:', OUTPUT_DIR)
print('Complete ZIP saved as:', archive_path)
print('Now start the interface with: py -m streamlit run app.py')
