# SALS-Net: Resolving Single-Cell T-Cell State Transitions and Pharmacological Remodeling via Optical Scattering

**SALS-Net** is an end-to-end, physics-informed deep learning framework designed to process raw, label-free single-shot Small-Angle Light Scattering (SALS) snapshots. By bypassing computationally heavy digital image reconstruction and avoiding the limitations of low-dimensional biophysical descriptors, SALS-Net extracts high-dimensional spatial representations directly from raw wave-diffraction patterns.

This platform delivers **instant, real-time predictions** to resolve continuous T-cell activation states, chronic dysfunction/exhaustion trajectories, and intermediate pharmacologically modulated phenotypes at single-cell resolution across independent human donor cohorts.

---

## Key Features

* **Label-Free & Non-Destructive:** Interrogates cells optically without exogenous biomarkers or destructive processing, keeping cells structurally pristine and viable for downstream molecular assays or patient infusion.
* **Rapid Network Optimization:** Converges in approximately **18 minutes** on a standard desktop workstation (profiled on an Intel Core i9-9900K CPU @ 3.60GHz).
* **Robust Preprocessing Pipeline:** Integrates custom, automated artifact-correction filters—including anisotropic vertical microfluidic line subtraction, algebraic circle-fitting alignment, and static camera-sensor spot inpainting.
* **Translational Compatibility:** Fully prepared to scale into an inline, real-time cell sorting/therapeutic separation device.

---

## Repository Structure

The workspace is organized as follows:

```bash
├── LICENSE                               # Terms of use (MIT License)
├── README.md                             # Repository documentation
├── SALS_Net__PreProcessing.py            # Phase 1: Image centering, alignment, & artifact cleaning
├── SALS_Net__Training.py                 # Phase 2: Model compilation & cross-donor optimization
├── SALS_Net__Testing.py                  # Phase 3: High-throughput batch inference & validation
│
# Trained Model Weights (Leave-One-Donor-Out Verification Schema)
├── best_sals_model_weights_Train_1.keras # Baseline training array (Set 1)
├── best_sals_model_weights_Train_2.keras # Baseline training array (Set 2)
├── best_sals_model_weights_Train_3.keras # Baseline training array (Set 3)
├── best_sals_model_weights_trained__1.keras # Cross-validation weights (Excluding Donor 1)
├── best_sals_model_weights_trained__2.keras # Cross-validation weights (Excluding Donor 2)
└── best_sals_model_weights_trained__3.keras # Cross-validation weights (Excluding Donor 3)

```

---

## Getting Started

### 1. Prerequisites & Installation

SALS-Net is built using Python and optimized for TensorFlow/Keras. Install the core dependencies via `pip`:

```bash
pip install tensorflow numpy opencv-python pillow matplotlib tqdm

```

### 2. Step 1: Preprocessing raw SALS snapshots

Run the unified preprocessing pipeline to align, crop, denoise, and normalize raw diffraction patterns into normalized, 300x300 grayscale matrices ready for SALS-Net.

```bash
python SALS_Net__PreProcessing.py

```

*This will launch a native file explorer window. Select your raw measurement folder. The program will output the standardized, clean images directly to a sibling folder named `<Folder_Name>_SALS_Preprocessed`.*

### 3. Step 2: Training the Network

Train the convolutional model using custom parameters, on-the-fly data augmentation, and early-stopping constraints.

```bash
python SALS_Net__Training.py

```

*The architecture contains approximately 1.85 million trainable parameters and typically converges in ~18 minutes on standard workstation setups.*

### 4. Step 3: Inference and Cross-Donor Validation

Deploy the trained models to perform high-throughput inference on independent validation datasets or to recreate cross-donor validations.

```bash
python SALS_Net__Testing.py

```

---

## Dataset Schema & Model Weights

The pre-trained weights included in this repository correspond to the **Leave-One-Donor-Out** validation schemes detailed in our manuscript:

* **`best_sals_model_weights_trained__1.keras`**: Trained on Donors 2 and 3; validated independently on **Donor 1**.
* **`best_sals_model_weights_trained__2.keras`**: Trained on Donors 1 and 3; validated independently on **Donor 2**.
* **`best_sals_model_weights_trained__3.keras`**: Trained on Donors 1 and 2; validated independently on **Donor 3**.

*The complete single-cell scattering image datasets supporting these validation schemas are permanently archived on Zenodo at [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.21299816).*

---

## License

This project is licensed under the MIT License.

---

## Citation

If you use SALS-Net, our preprocessing pipelines, or our datasets in your research, please cite our manuscript:

...
