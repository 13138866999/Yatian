### 🛠 Project Tech Stack

#### 1. Core Frameworks 

* **Deep Learning**: `PyTorch 2.x` ecosystem (including `torchvision`) for model construction and training.
* **Precision Training**: PyTorch AMP (Automatic Mixed Precision) & `GradScaler` for memory-efficient training.

#### 2. Computer Vision & Image Processing

* **Image Manipulation**: `OpenCV` (morphological operations for hair removal), `Pillow`, and `scikit-image` (RGB-LAB color space conversion).
* **Augmentation**: Standard `torchvision.transforms` (Resize, RandomResizedCrop, Normalize).
* **Feature Extraction**: Calculation of Individual Typology Angle (ITA) based on pixel-wise Lab color analysis.

#### 3. Model Architecture & Strategy

* **Backbone**: `ResNet50` (pretrained on ImageNet) with modified fully connected layers.
* **Optimization**: `Adam` optimizer with custom loss reweighting logic.
* **Validation**: K-Fold Cross-Validation (supporting `StratifiedKFold`, `GroupKFold`, and `StratifiedGroupKFold`) for robust evaluation.

#### 4. Data Science & Algorithms

* **Data Handling**: `Pandas` for metadata management; `NumPy` for high-performance numerical computing.
* **Mathematical Modeling**: `SciPy` (`gaussian_kde`) for Kernel Density Estimation to model skin tone distributions.
* **Fairness Metrics**: Implementation of Fidelity Similarity (FS), Wasserstein Distance (WD), and Patrick-Fisher Distance (PF) for distribution alignment.

#### 5. Infrastructure & Tools

* **Parallel Processing**: Python `concurrent.futures.ProcessPoolExecutor` for high-concurrency data preprocessing.
* **Progress Tracking**: `tqdm` for real-time training and processing feedback.
* **Version Control**: Standard Git workflow (implied by `.gitignore` config).
