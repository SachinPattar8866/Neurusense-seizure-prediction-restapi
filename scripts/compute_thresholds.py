import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESTAPI_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(RESTAPI_ROOT, '..'))

MODEL_METADATA_PATH = os.path.join(RESTAPI_ROOT, 'models', 'model_metadata.json')
TRAINING_IMAGES_DIR = os.path.join(WORKSPACE_ROOT, 'training', 'data', 'images')

if not os.path.exists(MODEL_METADATA_PATH):
    raise SystemExit('Model metadata not found')

with open(MODEL_METADATA_PATH, 'r') as f:
    metadata = json.load(f)

models_meta = metadata.get('models', {})
first_model = next(iter(models_meta.values()))
pre = first_model.get('preprocessing', {})
IMG_SIZE = int(pre.get('img_size', 128))
COLOR_MODE = pre.get('color_mode', 'rgb')
RESCALE = float(pre.get('rescale_factor', 1.0/255.0))

print(f"Using preprocessing img_size={IMG_SIZE}, color_mode={COLOR_MODE}, rescale={RESCALE}")

# Data generator for validation split
datagen = ImageDataGenerator(rescale=RESCALE, validation_split=0.2)

gen = datagen.flow_from_directory(
    TRAINING_IMAGES_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    color_mode=COLOR_MODE
)

label_map = gen.class_indices
print('class_indices:', label_map)

num_samples = gen.samples

# Load models
models = {}
for key, entry in models_meta.items():
    path = entry.get('exported_path')
    if os.path.exists(path):
        print('Loading', key, '...')
        models[key] = tf.keras.models.load_model(path)

# Collect per-sample seizure probabilities
true_labels = gen.classes
# class index for seizure
seizure_idx = label_map.get('seizure', 2)

probs = {k: [] for k in models.keys()}
for model_name, model in models.items():
    print('Predicting with', model_name)
    preds = model.predict(gen, verbose=0)
    preds = np.array(preds)
    if preds.ndim == 2:
        seizure_probs = preds[:, seizure_idx]
    else:
        seizure_probs = preds.flatten()
    probs[model_name] = seizure_probs

# Helper: compute precision, recall and thresholds using numpy
def precision_recall_vs_thresholds(y_true_bool, scores):
    # sort by score descending
    idx = np.argsort(-scores)
    y_sorted = y_true_bool[idx]
    scores_sorted = scores[idx]

    tp_cumsum = np.cumsum(y_sorted)
    fp_cumsum = np.cumsum(~y_sorted)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / np.sum(y_true_bool) if np.sum(y_true_bool) > 0 else np.zeros_like(tp_cumsum)

    # thresholds are the unique scores seen (descending)
    unique_scores, first_idx = np.unique(scores_sorted, return_index=True)
    thresholds = unique_scores
    return precision, recall, thresholds

# Compute thresholds and metrics
results = {}
for model_name, seizure_probs in probs.items():
    y_pos = (true_labels == seizure_idx)
    precision, recall, pr_thresh = precision_recall_vs_thresholds(y_pos, seizure_probs)

    f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
    best_thresh = float(pr_thresh[best_idx]) if len(pr_thresh) > 0 else 1.0
    best_f1 = float(f1_scores[best_idx]) if len(f1_scores) > 0 else 0.0

    # find threshold for 0.90 recall
    target_recall = 0.9
    th_for_target = 1.0
    for p, r, t in zip(precision, recall, pr_thresh):
        if r >= target_recall:
            th_for_target = float(t)
            break

    results[model_name] = {
        'best_f1_threshold': best_thresh,
        'best_f1': best_f1,
        'recall_0.9_threshold': th_for_target
    }

print('\nThreshold suggestions:')
for k, v in results.items():
    print(k, v)

# Ensemble thresholds: compute ensemble probs (weighted by F1 from metrics.csv)
metrics_csv = os.path.join(WORKSPACE_ROOT, 'training', 'results', 'metrics.csv')
weights = {}
if os.path.exists(metrics_csv):
    import csv
    with open(metrics_csv, newline='') as f:
        r = csv.DictReader(f)
        f1s = {}
        for row in r:
            f1s[row['model']] = float(row['f1_score'])
    total = 0.0
    for model_name in models.keys():
        if model_name == 'hybrid_best.h5' and 'hybrid_cnn_bilstm.h5' in f1s:
            fname = 'hybrid_cnn_bilstm.h5'
        elif model_name == 'cnn_best.h5' and 'cnn_baseline.h5' in f1s:
            fname = 'cnn_baseline.h5'
        else:
            fname = model_name
        f1 = f1s.get(fname, 0.5)
        weights[model_name] = f1
        total += f1
    for k in weights:
        weights[k] = weights[k] / total if total > 0 else 1.0 / len(weights)
else:
    n = len(models)
    weights = {k: 1.0 / n for k in models.keys()}

print('\nEnsemble weights:', weights)

ensemble = np.zeros(num_samples)
for model_name, seizure_probs in probs.items():
    ensemble += weights[model_name] * seizure_probs

precision, recall, pr_thresh = precision_recall_vs_thresholds(true_labels == seizure_idx, ensemble)

f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
best_thresh = float(pr_thresh[best_idx]) if len(pr_thresh) > 0 else 1.0

th_for_target = 1.0
for p, r, t in zip(precision, recall, pr_thresh):
    if r >= 0.9:
        th_for_target = float(t)
        break

results['ensemble'] = {
    'best_f1_threshold': best_thresh,
    'best_f1': float(f1_scores[best_idx]) if len(f1_scores) > 0 else 0.0,
    'recall_0.9_threshold': th_for_target
}

print('\nEnsemble results:', results['ensemble'])

# Save results
out_path = os.path.join(RESTAPI_ROOT, 'models', 'thresholds.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved thresholds to', out_path)
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESTAPI_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(RESTAPI_ROOT, '..'))

MODEL_METADATA_PATH = os.path.join(RESTAPI_ROOT, 'models', 'model_metadata.json')
TRAINING_IMAGES_DIR = os.path.join(WORKSPACE_ROOT, 'training', 'data', 'images')

if not os.path.exists(MODEL_METADATA_PATH):
    raise SystemExit('Model metadata not found')

with open(MODEL_METADATA_PATH, 'r') as f:
    metadata = json.load(f)

models_meta = metadata.get('models', {})
first_model = next(iter(models_meta.values()))
pre = first_model.get('preprocessing', {})
IMG_SIZE = int(pre.get('img_size', 128))
COLOR_MODE = pre.get('color_mode', 'rgb')
RESCALE = float(pre.get('rescale_factor', 1.0/255.0))

print(f"Using preprocessing img_size={IMG_SIZE}, color_mode={COLOR_MODE}, rescale={RESCALE}")

# Data generator for validation split
datagen = ImageDataGenerator(rescale=RESCALE, validation_split=0.2)

gen = datagen.flow_from_directory(
    TRAINING_IMAGES_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    color_mode=COLOR_MODE
)

label_map = gen.class_indices
print('class_indices:', label_map)

num_samples = gen.samples

# Load models
models = {}
for key, entry in models_meta.items():
    path = entry.get('exported_path')
    if os.path.exists(path):
        print('Loading', key, '...')
        models[key] = tf.keras.models.load_model(path)

# Collect per-sample seizure probabilities
true_labels = gen.classes
# class index for seizure
seizure_idx = label_map.get('seizure', 2)

probs = {k: [] for k in models.keys()}
for model_name, model in models.items():
    print('Predicting with', model_name)
    preds = model.predict(gen, verbose=0)
    preds = np.array(preds)
    if preds.ndim == 2:
        seizure_probs = preds[:, seizure_idx]
    else:
        seizure_probs = preds.flatten()
    probs[model_name] = seizure_probs

# Compute thresholds and metrics
results = {}
for model_name, seizure_probs in probs.items():
    precision, recall, pr_thresh = precision_recall_curve(true_labels == seizure_idx, seizure_probs)
    fpr, tpr, roc_thresh = roc_curve(true_labels == seizure_idx, seizure_probs)

    # find F1-maximizing threshold
    f1_scores = [(2 * p * r / (p + r)) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    best_idx = int(np.argmax(f1_scores))
    best_thresh = pr_thresh[best_idx] if best_idx < len(pr_thresh) else 1.0
    best_f1 = f1_scores[best_idx]

    # find threshold for 0.90 recall (or highest recall <=1)
    target_recall = 0.9
    # recall is non-increasing with threshold; we need threshold where recall >= target
    # precision_recall_curve returns thresholds of length n-1; align accordingly
    th_for_target = None
    for p, r, t in zip(precision, recall, np.append(pr_thresh, 1.0)):
        if r >= target_recall:
            th_for_target = t
            break
    if th_for_target is None:
        th_for_target = 1.0

    results[model_name] = {
        'best_f1_threshold': float(best_thresh),
        'best_f1': float(best_f1),
        'recall_0.9_threshold': float(th_for_target)
    }

print('\nThreshold suggestions:')
for k, v in results.items():
    print(k, v)

# Ensemble thresholds: compute ensemble probs (weighted by F1 from metrics.csv)
# Read training metrics for F1 weights
metrics_csv = os.path.join(WORKSPACE_ROOT, 'training', 'results', 'metrics.csv')
weights = {}
if os.path.exists(metrics_csv):
    import csv
    with open(metrics_csv, newline='') as f:
        r = csv.DictReader(f)
        f1s = {}
        for row in r:
            f1s[row['model']] = float(row['f1_score'])
    # map filenames to model keys in models
    total = 0.0
    for model_name in models.keys():
        # map model_name like 'hybrid_best.h5' to training filename 'hybrid_cnn_bilstm.h5' if needed
        key = model_name
        if model_name == 'hybrid_best.h5' and 'hybrid_cnn_bilstm.h5' in f1s:
            fname = 'hybrid_cnn_bilstm.h5'
        elif model_name == 'cnn_best.h5' and 'cnn_baseline.h5' in f1s:
            fname = 'cnn_baseline.h5'
        else:
            fname = model_name
        f1 = f1s.get(fname, 0.5)
        weights[model_name] = f1
        total += f1
    # normalize
    for k in weights:
        weights[k] = weights[k] / total if total > 0 else 1.0 / len(weights)
else:
    # equal weights
    n = len(models)
    weights = {k: 1.0 / n for k in models.keys()}

print('\nEnsemble weights:', weights)

# ensemble seizure probs
ensemble = np.zeros(num_samples)
for model_name, seizure_probs in probs.items():
    ensemble += weights[model_name] * seizure_probs

# compute ensemble thresholds with same method
precision, recall, pr_thresh = precision_recall_curve(true_labels == seizure_idx, ensemble)
f1_scores = [(2 * p * r / (p + r)) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
best_idx = int(np.argmax(f1_scores))
best_thresh = pr_thresh[best_idx] if best_idx < len(pr_thresh) else 1.0

th_for_target = None
for p, r, t in zip(precision, recall, np.append(pr_thresh, 1.0)):
    if r >= 0.9:
        th_for_target = t
        break
if th_for_target is None:
    th_for_target = 1.0

results['ensemble'] = {
    'best_f1_threshold': float(best_thresh),
    'best_f1': float(f1_scores[best_idx]),
    'recall_0.9_threshold': float(th_for_target)
}

print('\nEnsemble results:', results['ensemble'])

# Save results
out_path = os.path.join(RESTAPI_ROOT, 'models', 'thresholds.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved thresholds to', out_path)
