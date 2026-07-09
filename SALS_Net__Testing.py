"""
SALS-Net: High-Throughput Inference & Phenotypic Vector Extraction Pipeline
===========================================================================
Author: David et al.
Year: 2026
Description: This script handles fast, GPU-accelerated batch inference on label-free
             single-cell scattering datasets. It loads a trained SALS-Net model,
             streams grayscale images through a native TF Graph normalization pipeline, 
             and exports precise single-cell classification probability matrices (.csv).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. COMPUTE HARDWARE & VRAM METRIC INITIALIZATION
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Active Hardware Accelerator Detected: {gpus[0].name}")
    for gpu in gpus:
        # Prevent TensorFlow from locking the entirety of host VRAM arbitrarily
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU hardware detected. Falling back to host CPU multi-threading configuration.")

# ==============================================================================
# 2. RUNTIME PARAMETERS & DIRECTORY RESOLUTION
# ==============================================================================
# Repository-relative directory setups for clean cloning across varying environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

#MODEL_PATH = os.path.join(BASE_DIR, 'best_sals_model_weights__trained__2.keras')
#INPUT_FOLDER = os.path.join(BASE_DIR, 'Data', 'Test_Dataset')

# Safe absolute path overrides (uncomment if targeting local machine hardware)
MODEL_PATH = 'best_sals_model_weights__trained__2.keras' 
INPUT_FOLDER = r'C:\Python\David\T_2026\Test\Results\Patient_2025_Test_2_T'

DIM = 150
BATCH_SIZE = 64 
CLASS_NAMES = ['acute stim.', 'chronic stim.']

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"SALS-Net weight files missing at: {MODEL_PATH}. "
                            "Please ensure weights are generated or downloaded before testing execution.")

if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(f"Source evaluation directory missing at: {INPUT_FOLDER}. "
                            "Please construct structured subdirectories corresponding to treatment labels.")

# ==============================================================================
# 3. HIGH-SPEED STREAMING PIPELINE WITH TF-GRAPH MIN-MAX OPTIMIZATION
# ==============================================================================
print(f"Loading compiled Keras model signature from: {MODEL_PATH}...")
model = load_model(MODEL_PATH)

print("Spinning up high-performance TensorFlow dataset stream pipelines...")
dataset = tf.keras.utils.image_dataset_from_directory(
    INPUT_FOLDER,
    label_mode=None,       # No categorical labels for raw experimental screens
    image_size=(DIM, DIM),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False          # CRITICAL: Keep files sequentially stable to ensure name alignment
)

# Extract file details before structural tensor transformation loops take place
file_paths = dataset.file_paths
treatment_labels = [os.path.basename(os.path.dirname(p)) for p in file_paths]
filenames = [os.path.basename(p) for p in file_paths]

def tf_sample_wise_min_max(img):
    """
    Executes sample-wise mathematical normalization directly inside the 
    TensorFlow computational graph to leverage localized GPU multi-processing.
    """
    img = tf.cast(img, tf.float32)
    img_min = tf.reduce_min(img, axis=[0, 1, 2], keepdims=True)
    img_max = tf.reduce_max(img, axis=[0, 1, 2], keepdims=True)
    img_normalized = (img - img_min) / (img_max - img_min + 1e-7)
    return img_normalized

# Map graph operations and configure optimized buffer memory spaces
dataset = dataset.map(tf_sample_wise_min_max).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# 4. PARALLELISED BATCH INFERENCE ROUTING
# ==============================================================================
print(f"\nProcessing forward evaluation arrays for {len(file_paths)} single-cell elements...")
all_preds = model.predict(dataset, verbose=1) 

# ==============================================================================
# 5. DATA AGGREGATION & VECTOR TRANSFORMATION
# ==============================================================================
acute_probs = all_preds[:, 0]
chronic_probs = all_preds[:, 1]
class_indices = np.argmax(all_preds, axis=1)

results_data = {
    'Treatment': treatment_labels,
    'Filename': filenames,
    'Predicted_Class': [CLASS_NAMES[i] for i in class_indices],
    'Acute_Prob': np.round(acute_probs, 4),
    'Chronic_Prob': np.round(chronic_probs, 4),
    'Verdict': ["More Acute" if a > c else "More Chronic" for a, c in zip(acute_probs, chronic_probs)]
}

df_full = pd.DataFrame(results_data)
df_activated = df_full.copy() # Keeps absolute synchronization with downstream dashboard code

# ==============================================================================
# 6. CSV DATA ARCHIVING & EXPORT SUMMARIES
# ==============================================================================
out_full = os.path.join(BASE_DIR, "Full_Predictions.csv")
out_activated = os.path.join(BASE_DIR, "Activated_Only_Results.csv")

df_full.to_csv(out_full, index=False)
df_activated.to_csv(out_activated, index=False)

print("\n" + "=" * 60)
print("             SALS-NET PIPELINE INFERENCE SUMMARY             ")
print("=" * 60)
summary = df_activated.groupby(['Treatment', 'Verdict']).size().unstack(fill_value=0)
print(summary)

print("\n--- PHENOTYPIC RESCUE EFFICIENCY (Acute % inside activated pools) ---")
try:
    efficiency = df_activated.groupby('Treatment')['Verdict'].value_counts(normalize=True).unstack() * 100
    print(efficiency.round(2).astype(str) + " %")
except Exception:
    efficiency = df_activated.groupby(['Treatment', 'Verdict']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).unstack(fill_value=0)
    print(efficiency.round(2).astype(str) + " %")
print("=" * 60)

print(f"\nInference operation concluded successfully.")
print(f"    Raw matrix tracking tables saved to:\n    -> {out_full}\n    -> {out_activated}")