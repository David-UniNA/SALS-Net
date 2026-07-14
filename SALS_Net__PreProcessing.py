"""
SALS-Net: Unified Optical Preprocessing & Artifact Rectification Pipeline
===========================================================================
Author: David et al.
Year: 2026
Description: A highly optimized, end-to-end preprocessing pipeline for raw
             single-cell Small-Angle Light Scattering (SALS) patterns.
             This script automates:
             1. Geometric centering & alignment (via circular fitting).
             2. Cropping, smoothing, and downsampling to CNN target dimensions (300x300).
             3. Anisotropic vertical microfluidic shadow profile subtraction.
             4. Static hardware/dead-pixel spot masking and inpainting.
             5. Real-time central beam-stop enforcement and normalization.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# ==============================================================================
# 1. GLOBAL PARAMETER CONFIGURATION
# ==============================================================================
TARGET_IMAGE_SIZE = 300      # Target dimensions for SALS-Net input layer
ALIGN_CANVAS_SIZE = 700      # Internal processing canvas size
SAFETY_MASK_RADIUS = 55      # Central beam-stop radius on the 700x700 canvas (scales to ~26 on 300x300)
NATIVE_CENTER_X = 150        # Center point of target 300x300 image
NATIVE_CENTER_Y = 150        # Center point of target 300x300 image

# ==============================================================================
# 2. FILE AND SYSTEM I/O UTILITIES
# ==============================================================================
def select_folder_via_explorer():
    """Launches a native OS directory selector."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_dir = filedialog.askdirectory(title="Select Raw SALS Measurement Folder")
    return Path(selected_dir) if selected_dir else None

def load_robust_monochrome(path):
    """Loads an image reliably as grayscale, handling multichannel arrays."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) >= 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

# ==============================================================================
# 3. STAGE 1: GEOMETRIC CENTER REGISTRATION
# ==============================================================================
def compute_folder_average_image(img_paths):
    """Pass 1: Accumulates and averages all images to compute the optical baseline."""
    first_img = load_robust_monochrome(img_paths[0])
    if first_img is None:
        return None
    h, w = first_img.shape
    
    master_accumulator = np.zeros((h, w), dtype=np.float32)
    valid_count = 0
    
    for path in img_paths:
        img = load_robust_monochrome(path)
        if img is not None and img.shape == (h, w):
            master_accumulator += img
            valid_count += 1
            
    if valid_count == 0:
        return None
        
    master_average = master_accumulator / valid_count
    return master_average.astype(np.uint8)

def fit_circle_to_ring_edges(img):
    """Pass 2: Fits an algebraic circle to locate the exact diffraction center."""
    h, w = img.shape
    crop_margin = 150
    roi = img[crop_margin:h-crop_margin, crop_margin:w-crop_margin]
    
    _, max_val, _, _ = cv2.minMaxLoc(roi)
    binary_ring = np.where(roi >= (max_val * 0.55), 255, 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    all_edge_pts = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            for pt in c:
                global_x = pt[0][0] + crop_margin
                global_y = pt[0][1] + crop_margin
                all_edge_pts.append((global_x, global_y))
                
    if len(all_edge_pts) < 10:
        return w / 2.0, h / 2.0
        
    pts = np.array(all_edge_pts, dtype=np.float32)
    X = pts[:, 0]
    Y = pts[:, 1]
    
    A = np.zeros((len(X), 3))
    A[:, 0] = X
    A[:, 1] = Y
    A[:, 2] = 1
    B = X**2 + Y**2
    
    try:
        result, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        xc = result[0] / 2.0
        yc = result[1] / 2.0
        return xc, yc
    except np.linalg.LinAlgError:
        return w / 2.0, h / 2.0

def align_and_mask_frame(raw_img, master_x, master_y, target_center=(350, 350), mask_radius=55):
    """Pass 3: Centers the diffraction pattern and masks the central beam-stop."""
    h, w = raw_img.shape
    dx = target_center[0] - master_x
    dy = target_center[1] - master_y
    
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_img = cv2.warpAffine(
        raw_img, 
        translation_matrix, 
        (w, h), 
        flags=cv2.INTER_CUBIC
    )
    
    cv2.circle(aligned_img, target_center, mask_radius, 0, -1)
    return aligned_img

# ==============================================================================
# 4. STAGE 2: ARTIFACT REMOVAL & PRECISION CLEANING
# ==============================================================================
def clean_anisotropic_background(img):
    """Surgically subtracts vertical microfluidic channel line shadows on far margins."""
    h, w = img.shape
    img_float = img.astype(np.float32)
    
    # Left Flank Profile (cols 0-45)
    left_profile = np.median(img_float[:, 0:45], axis=1)
    left_profile = cv2.GaussianBlur(left_profile, (1, 5), 0).flatten()
    
    # Right Flank Profile (cols 255-300)
    right_profile = np.median(img_float[:, 255:300], axis=1)
    right_profile = cv2.GaussianBlur(right_profile, (1, 5), 0).flatten()
    
    correction_map = np.zeros_like(img_float)
    
    for y in range(h):
        # Apply subtraction at far left edge, blending out by column 50
        for x in range(0, 50):
            weight = (50 - x) / 50.0
            correction_map[y, x] = (left_profile[y] - np.min(left_profile)) * weight
            
        # Apply subtraction at far right edge, blending out by column 250
        for x in range(250, w):
            weight = (x - 250) / 50.0
            correction_map[y, x] = (right_profile[y] - np.min(right_profile)) * weight

    cleaned = cv2.subtract(img_float, correction_map)
    return np.clip(cleaned, 0, 255).astype(np.uint8)

def generate_static_hardware_mask(img_paths, master_x, master_y, inner_r=23, outer_r=65):
    """Generates a static mask for horizontal sensor lines and dead pixels on aligned samples."""
    cx, cy = int(NATIVE_CENTER_X), int(NATIVE_CENTER_Y)
    sample_paths = img_paths[::max(1, len(img_paths) // 30)][:30]
    
    accumulated = np.zeros((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), dtype=np.float64)
    count = 0
    
    for p in sample_paths:
        raw_img = load_robust_monochrome(p)
        if raw_img is None:
            continue
        
        # Bring sample into centered alignment
        aligned = align_and_mask_frame(raw_img, master_x, master_y, (350, 350), SAFETY_MASK_RADIUS)
        cropped = aligned[50:650, 50:650]
        resized = cv2.resize(cropped, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        
        accumulated += resized
        count += 1
            
    master_avg = (accumulated / count).astype(np.uint8)
    h, w = master_avg.shape
    
    y_indices, x_indices = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
    ring_zone_mask = (dist_from_center >= inner_r) & (dist_from_center <= outer_r)
    
    global_median = np.median(master_avg[ring_zone_mask])
    hardware_spots = (master_avg < (global_median * 0.45)) & ring_zone_mask
    
    hardware_mask = np.zeros_like(master_avg, dtype=np.uint8)
    hardware_mask[hardware_spots] = 255
    
    # Target sharp horizontal sensor line artifacts (Row 194)
    line_y = 194 
    hardware_mask[line_y-1:line_y+2, 0:105] = 255     # Left segment
    hardware_mask[line_y-3:line_y+5, 195:300] = 255   # Right bleeding segment
    
    cv2.circle(hardware_mask, (cx, cy), int(inner_r), 0, -1)
    return hardware_mask

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION ENGINE
# ==============================================================================
def run_unified_pipeline():
    src_root = select_folder_via_explorer()
    if src_root is None or str(src_root) == ".":
        print(" Pipeline execution canceled by user.")
        return
        
    dest_root = src_root.parent / f"{src_root.name}_SALS_Preprocessed"
    
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    img_paths = sorted([p for ext in extensions for p in src_root.rglob(ext)])
    
    if not img_paths:
        print(f" Error: No valid image files found in: {src_root}")
        return
        
    print(f" Found {len(img_paths)} frames. Initiating Stage 1...")
    
    # Stage 1: Align & compute Master Folder Average Coordinates
    print("    -> Computing Folder Average baseline...")
    avg_master_img = compute_folder_average_image(img_paths)
    if avg_master_img is None:
        print(" Error: Could not compile average folder canvas.")
        return
        
    print("    -> Calculating optical center registration via circular boundary fitting...")
    master_x, master_y = fit_circle_to_ring_edges(avg_master_img)
    print(f"     Calibration Locked: Center X = {master_x:.2f}, Center Y = {master_y:.2f}")
    
    # Stage 2: Generate Static Hardware Defect Mask
    print("\n Initiating Stage 2 (Hardware Defect Profiling)...")
    print("    -> Modeling static camera sensor artifacts and dead pixel vectors...")
    static_mask = generate_static_hardware_mask(
        img_paths, 
        master_x, 
        master_y, 
        inner_r=23, 
        outer_r=65
    )
    
    # Stage 3: End-to-End Image Processing Loop
    print("\n Commencing parallelized batch preprocessing transformations...")
    print("-" * 75)
    print(f"Source Folder:      {src_root}")
    print(f"Destination Folder: {dest_root}")
    print(f"Output Target:      {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE} pixels")
    print("-" * 75)
    
    processed_count = 0
    
    for img_path in img_paths:
        raw_img = load_robust_monochrome(img_path)
        if raw_img is None:
            continue
            
        # 1. Optical Centering Alignment
        aligned_frame = align_and_mask_frame(
            raw_img, 
            master_x, 
            master_y, 
            target_center=(350, 350), 
            mask_radius=SAFETY_MASK_RADIUS
        )
        
        # 2. Central Crop (600x600 around center) & Downsampling to target 300x300
        cropped = aligned_frame[50:650, 50:650]
        smoothed = cv2.GaussianBlur(cropped, (3, 3), 0)
        resized = cv2.resize(smoothed, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # 3. Microfluidic Vertical Shadow Subtraction
        flank_cleaned = clean_anisotropic_background(resized)
        
        # 4. Inpaint localized sensor rows & spots
        inpainted = cv2.inpaint(flank_cleaned, static_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        
        # 5. Enforce crisp, dark solid beam-stop center circle
        cv2.circle(inpainted, (NATIVE_CENTER_X, NATIVE_CENTER_Y), 26, 0, -1)
        
        # 6. Sample-wise Min-Max scale normalization
        min_v, max_v = np.min(inpainted), np.max(inpainted)
        if max_v > min_v:
            final_output = ((inpainted - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            final_output = np.zeros_like(inpainted, dtype=np.uint8)
            
        # 7. Mirror directory hierarchy inside processed folder structure
        relative_path = img_path.relative_to(src_root)
        save_path = dest_root / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path), final_output)
        processed_count += 1
        
    print("-" * 75)
    print(f"    SUCCESS: Preprocessing and artifact correction loop complete.")
    print(f"    Processed and standard-formatted {processed_count} images.")
    print(f"    SALS-Net compatible output saved to: {dest_root}")

if __name__ == "__main__":
    run_unified_pipeline()