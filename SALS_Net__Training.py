"""
SALS-Net: Multi-Scale Inception Network for T-Cell Activation Classification
===========================================================================
Author: David et al.
Year: 2026
Description: This script contains the end-to-end training pipeline for SALS-Net.
             It features a high-speed GPU-accelerated dataset implementation, 
             a multi-scale parallelized Inception network architecture, 
             and automated generation of training metrics figures.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import keras
from keras import layers, models, callbacks
# Force Keras backend to PyTorch before importing Keras modules
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# 1. GLOBAL CONFIGURATION & DATA CACHING RUNTIME
# ==============================================================================
RESET_CACHE = True          # Set to True to force deletion/overwriting of old compressed numpy data arrays

# Use relative paths by default for repository clean-cloning compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
CACHE_DIR = os.path.join(BASE_DIR, "Data_Cache")
TRAIN_DIR = os.path.join(BASE_DIR, "Data", "Train_Patients")
VAL_DIR   = os.path.join(BASE_DIR, "Data", "Validation_Patients")

# Safe overrides if absolute structures are locally required
# CACHE_DIR = r"C:\Python\David\T_2026\Test\Results"
# TRAIN_DIR = r"C:\Python\David\T_2026\Test\Results\Train_Patients_2"
# VAL_DIR   = r"C:\Python\David\T_2026\Test\Results\Patient_2025_Test_2"

os.makedirs(CACHE_DIR, exist_ok=True)

# Define file targets for the pre-processed matrix cache
X_TRAIN_SAVE = os.path.join(CACHE_DIR, "x_train_cache.npy")
Y_TRAIN_SAVE = os.path.join(CACHE_DIR, "y_train_cache.npy")
X_VAL_SAVE   = os.path.join(CACHE_DIR, "x_val_cache.npy")
Y_VAL_SAVE   = os.path.join(CACHE_DIR, "y_val_cache.npy")

if RESET_CACHE:
    print("\n" + "!" * 60)
    print("RESET_CACHE=True: Invalidating and purging stale cache maps...")
    print("!" * 60)
    
    cache_files = [X_TRAIN_SAVE, Y_TRAIN_SAVE, X_VAL_SAVE, Y_VAL_SAVE]
    deleted_count = 0
    for file_path in cache_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  [✔] Purged matrix: {os.path.basename(file_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {os.path.basename(file_path)}. Error: {e}")
        else:
            print(f"  [·] {os.path.basename(file_path)} was not present/already empty.")
            
    print(f"\n[✔] Cache cleanup complete. Removed {deleted_count} stale cache assets.")
    print("    Pipeline will reconstruct feature matrices from raw assets.\n" + "=" * 50)
else:
    print("\n[ℹ] Cache preservation active. Initializing high-speed vector array loading.\n")

# ==============================================================================
# 2. HARDWARE VERIFICATION & COMPUTE ENGINE SETUP
# ==============================================================================
print("\n" + "=" * 50)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware Compute Verification: PIPELINE RUNNING ON -> [{device.upper()}]")
if device == "cuda":
    print(f"Active Graphics Processing Unit: {torch.cuda.get_device_name(0)}")
print("=" * 50 + "\n")

# Network Input/Training parameters
IMG_HEIGHT = 150  
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 50

# ==============================================================================
# 3. HIGH-PERFORMANCE IN-MEMORY CACHE LOADER
# ==============================================================================
def build_or_load_matrix_cache(directory, x_path, y_path):
    """
    Checks for the presence of compressed numpy arrays. If missing, aggregates raw 
    image subdirectories into consolidated matrix blocks to optimize pipeline I/O.
    
    Parameters:
        directory (str): Path containing subfolders representing target classes.
        x_path (str): File path destination for the sample image array.
        y_path (str): File path destination for the target label array.
        
    Returns:
        tuple: (x_matrix, y_matrix) holding unified training/validation tensors.
    """
    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"--> Found cached array matrices for: {os.path.basename(directory)}.")
        return np.load(x_path), np.load(y_path)
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Source directory path not found: {directory}. "
                                "Please verify local directory structures or environment setups.")
        
    print(f"--> Building compressed array matrices for raw source: {os.path.basename(directory)}...")
    ds = keras.utils.image_dataset_from_directory(
        directory, 
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        batch_size=128, 
        color_mode='grayscale', 
        shuffle=False
    )
    
    imgs, lbls = [], []
    for im, lb in ds:
        imgs.append(im.numpy().astype(np.uint8))
        lbls.append(lb.numpy())
        
    x_mat, y_mat = np.concatenate(imgs, axis=0), np.concatenate(lbls, axis=0)
    np.save(x_path, x_mat)
    np.save(y_path, y_mat)
    return x_mat, y_mat

# Aggregate and load datasets into memory matrices
x_train_raw, y_train_raw = build_or_load_matrix_cache(TRAIN_DIR, X_TRAIN_SAVE, Y_TRAIN_SAVE)
x_val_raw, y_val_raw = build_or_load_matrix_cache(VAL_DIR, X_VAL_SAVE, Y_VAL_SAVE)

x_train_float = x_train_raw.astype(np.float32)
x_val_float = x_val_raw.astype(np.float32)

# --- Sample-Wise Min-Max Normalization (0.0 - 1.0) ---
print("Executing sample-wise range scaling optimization on Training dataset...")
for i in range(x_train_float.shape[0]):
    img_min = x_train_float[i].min()
    img_max = x_train_float[i].max()
    x_train_float[i] = (x_train_float[i] - img_min) / (img_max - img_min + 1e-7)

print("Executing sample-wise range scaling optimization on Validation dataset...")
for i in range(x_val_float.shape[0]):
    img_min = x_val_float[i].min()
    img_max = x_val_float[i].max()
    x_val_float[i] = (x_val_float[i] - img_min) / (img_max - img_min + 1e-7)

# Cast arrays directly to PyTorch tensors locked on GPU VRAM for runtime acceleration
x_train = torch.tensor(x_train_float, dtype=torch.float32).cuda()
x_val = torch.tensor(x_val_float, dtype=torch.float32).cuda()
y_train = torch.tensor(y_train_raw, dtype=torch.long).cuda()
y_val = torch.tensor(y_val_raw, dtype=torch.long).cuda()

print(f"VRAM Blocks Locked: Train Shape = {x_train.shape} | Val Shape = {x_val.shape}")
print(f"Normalization Sanity Check: Train Range [{x_train.min().item():.2f}, {x_train.max().item():.2f}]")

# ==============================================================================
# 4. IN-VRAM DATASET DEFINITION & REAL-TIME AUGMENTATION
# ==============================================================================
class HighSpeedGpuDataset:
    """
    Custom PyTorch-based Dataset wrapper optimized to compute transformations and
    spatial data augmentations directly inside GPU memory space during training loops.
    """
    def __init__(self, x, y, augment=False):
        self.x = x
        self.y = y
        self.augment = augment
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        img = self.x[idx] 
        
        if self.augment:
            # Transpose to Channel-First format (C, H, W) for torchvision transforms
            img = img.permute(2, 0, 1)
            
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
                
            # Random 360-degree rotational variation
            angle = random.uniform(-180, 180)
            img = TF.rotate(img, angle, fill=[0])
            
            # Revert back to Channel-Last format (H, W, C) for Keras consumption
            img = img.permute(1, 2, 0)
            
        return img, self.y[idx]

# Instantiating active memory streaming loaders
train_loader = DataLoader(
    HighSpeedGpuDataset(x_train, y_train, augment=True), 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    generator=torch.Generator()
)
val_loader = DataLoader(
    HighSpeedGpuDataset(x_val, y_val, augment=False), 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# ==============================================================================
# 5. MULTI-SCALE INCEPTION NETWORK ARCHITECTURE (SALS-Net)
# ==============================================================================
def inception_module(x, filters, reg):
    """
    Constructs a parallel, multi-scale feature extraction block mapping varied receptive fields.
    Includes 1x1, 3x3, and 5x5 spatial convolution kernels alongside structural dropout regularization.
    """
    # 1x1 Convolutions path
    path_1x1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=reg)(x)
    
    # 3x3 Convolutions path
    path_3x3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=reg)(x)
    path_3x3 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(path_3x3)
    
    # 5x5 Convolutions path
    path_5x5 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=reg)(x)
    path_5x5 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu', kernel_regularizer=reg)(path_5x5)
    
    # Concatenate parallel structural channels
    out = layers.Concatenate()([path_1x1, path_3x3, path_5x5])
    out = layers.BatchNormalization()(out)
    out = layers.SpatialDropout2D(0.15)(out)
    return out

def build_multiscale_network(num_classes=5):
    """
    Compiles global deep feature representations mapping integrated regularized multi-scale 
    Inception modules for cell categorical predictions.
    """
    l2_reg = keras.regularizers.l2(1e-3) 
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Initial Stem Block
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Functional Blocks Flow
    x = inception_module(x, 32, l2_reg)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = inception_module(x, 64, l2_reg)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully-Connected Dense Block
    x = layers.Dense(128, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.6)(x) 
    
    outputs = layers.Dense(num_classes, activation='softmax', name="Classification_Layer")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="SALS_Inception_Network")

model = build_multiscale_network(num_classes=5)

# ==============================================================================
# 6. PIPELINE OPTIMIZATION & TRAINING MECHANICS
# ==============================================================================
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=6e-4, weight_decay=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks_list = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(
        filepath='best_sals_model_weights.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
]

print("\nLaunching SALS-Net Optimization Training Loop...")
history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    verbose=1
)

# ==============================================================================
# 7. METRICS CONVERGENCE SUMMARY & FIGURE GENERATION
# ==============================================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

best_epoch_idx = np.argmax(val_acc)
best_epoch = best_epoch_idx + 1
best_val_acc = val_acc[best_epoch_idx]
corresponding_val_loss = val_loss[best_epoch_idx]

# Console summary logs
print("\n" + "=" * 50)
print("             METRICS SUMMARY            ")
print("=" * 50)
print(f"Total Epochs Processed:     {len(acc)}")
print(f"Optimal Convergence Epoch:  Halting point restored to Epoch {best_epoch}")
print(f"Max Validation Accuracy:    {best_val_acc * 100:.2f}%")
print(f"Validation Loss at Target:  {corresponding_val_loss:.4f}")
print(f"Final Training Accuracy:    {acc[-1] * 100:.2f}%")
print(f"Final Training Loss:        {loss[-1]:.4f}")
print("=" * 50 + "\n")

# Setup typography standard rules
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DeVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# Panel (a): Classification Accuracy Metrics
ax1.plot(epochs_range, acc, label='Training Set', color='#1f77b4', linestyle='-', linewidth=2)
ax1.plot(epochs_range, val_acc, label='Validation Set', color='#ff7f0e', linestyle='--', linewidth=2)
ax1.axvline(x=best_epoch, color='#d62728', linestyle=':', label=f'Best Model (Ep. {best_epoch})', alpha=0.8)
ax1.set_title('(a) Multi-Scale Network Classification Accuracy')
ax1.set_xlabel('Training Epochs')
ax1.set_ylabel('Accuracy Proportion')
ax1.set_ylim(0.0, 1.0)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

# Panel (b): Structural Cross-Entropy Loss Curve Trajectories
ax2.plot(epochs_range, loss, color='#1f77b4', linestyle='-', linewidth=2)
ax2.plot(epochs_range, val_loss, color='#ff7f0e', linestyle='--', linewidth=2)
ax2.axvline(x=best_epoch, color='#d62728', linestyle=':', alpha=0.8)
ax2.set_title('(b) Categorical Cross-Entropy Loss Trajectory')
ax2.set_xlabel('Training Epochs')
ax2.set_ylabel('Loss Metric Value')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Export paths for high-fidelity reproduction vector spaces
output_png = 'sals_inception_training_metrics.png'
plt.savefig(output_png, bbox_inches='tight', dpi=300)
print(f"Figure metrics exported successfully ({output_png}).")

plt.show()