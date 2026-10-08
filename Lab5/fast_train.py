import joblib
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import kagglehub

# Fast bypass script to retrain the exact chosen model with the normal images
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(kagglehub.dataset_download('sovitrath/neu-steel-surface-defect-detect-trainvalid-split'))
NORMAL_IMAGES_DIR = BASE_DIR / 'normal_images'
OUTPUT_DIR = BASE_DIR / 'lab05_results'

def find_split(root):
    root = Path(root)
    for parent in [root, *root.rglob('*')]:
        if not parent.is_dir():
            continue
        for train_name, valid_name in [('train_images', 'valid_images'), ('train/images', 'valid/images')]:
            train_dir, valid_dir = parent / train_name, parent / valid_name
            if train_dir.is_dir() and valid_dir.is_dir():
                return train_dir, valid_dir
    return None, None

def label_from_path(path):
    import re
    stem = re.sub(r'[_-]\d+.*$', '', path.stem.lower())
    key = re.sub(r'[^a-z0-9]+', '_', stem).strip('_')
    return key

train_dir, valid_dir = find_split(DATASET_ROOT)
train_paths = sorted(p for p in train_dir.rglob('*') if p.is_file())
normal_paths = sorted(p for p in NORMAL_IMAGES_DIR.rglob('*') if p.is_file())

y_train = [label_from_path(p) for p in train_paths]
y_train += ['normal'] * len(normal_paths)
train_paths += normal_paths

def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L').resize((80, 80), Image.Resampling.BILINEAR))

print("Loading images...")
train_images = np.stack([read_gray(p) for p in train_paths])

print("Computing HOG features...")
rows = []
for image in train_images:
    h = hog(image, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
            block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
    rows.append(h)
X_train = np.asarray(rows, dtype=np.float32)

print("Training model...")
estimator = SVC(kernel='rbf', C=3.0, gamma='scale', class_weight='balanced')
pipeline = make_pipeline(StandardScaler(with_mean=False), estimator)
model = CalibratedClassifierCV(pipeline, cv=3, method='sigmoid')
model.fit(X_train, y_train)

classes = sorted(list(set(y_train)))

model_bundle = {
    'model': model, 'classes': classes, 'image_size': (80, 80),
    'hog_cell_size': 8, 'hog_orientations': 12,
    'model_name': 'RBF SVM C=3',
}
print("Saving model...")
joblib.dump(model_bundle, OUTPUT_DIR / 'inspection_model.joblib', compress=3)
joblib.dump(model_bundle, BASE_DIR / 'inspection_model.joblib', compress=3)
print("Done! Model saved directly to BASE_DIR.")
