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
    print(f"Model metadata not found at {MODEL_METADATA_PATH}")
    raise SystemExit(1)

if not os.path.isdir(TRAINING_IMAGES_DIR):
    print(f"Training images folder not found at {TRAINING_IMAGES_DIR}")
    raise SystemExit(1)

with open(MODEL_METADATA_PATH, 'r') as f:
    metadata = json.load(f)

models_meta = metadata.get('models', {})
if not models_meta:
    print('No models found in metadata')
    raise SystemExit(1)

# Use preprocessing config from one of the models (they should match)
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

num_samples = gen.samples
num_classes = gen.num_classes
print(f"Validation samples: {num_samples}, classes: {num_classes}, class_indices: {gen.class_indices}")

if num_samples == 0:
    print('No validation samples found - cannot evaluate.')
    raise SystemExit(0)

results = {}

for model_key, model_entry in models_meta.items():
    model_path = model_entry.get('exported_path') or model_entry.get('model_filename')
    if not os.path.exists(model_path):
        print(f"Model file not found for {model_key}: {model_path}")
        continue

    print(f"\nLoading model {model_key} from {model_path}")
    model = tf.keras.models.load_model(model_path)

    print('Running predictions on validation set...')
    preds = model.predict(gen, verbose=0)
    preds = np.array(preds)

    if preds.ndim == 1:
        # binary/single output
        pred_labels = (preds > 0.5).astype(int)
    else:
        pred_labels = np.argmax(preds, axis=1)

    true_labels = gen.classes

    acc = float(np.mean(pred_labels == true_labels))

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1

    results[model_key] = {'accuracy': acc, 'confusion_matrix': cm.tolist()}
    print(f"Model {model_key} accuracy: {acc:.4f}")
    print('Confusion matrix:')
    print(cm)

print('\nEvaluation complete.')

# Print summary
for k, v in results.items():
    print(f"\n{k}: accuracy={v['accuracy']:.4f}")
    print('confusion_matrix:')
    for row in v['confusion_matrix']:
        print(row)
