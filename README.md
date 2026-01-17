# FALCON: Federated Learning Framework

FALCON is a federated learning framework designed for medical image classification and other computer vision tasks. It implements a multi-stage training pipeline with feature encoding, local classifier training, synthetic data generation, and global model aggregation.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
- [Citation](#citation)
- [License](#license)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.4.1+
- torchvision 0.19.1+
- Other dependencies: numpy, Pillow, scikit-learn, tqdm, transformers

### Setup

```bash
git clone <repository-url>
cd FALCON
pip install -r requirements.txt
```

## Project Structure

```
FALCON/
├── config.py                 # Configuration file for datasets, models, paths
├── dataset.py                # Dataset loading and preprocessing utilities
├── utils.py                  # Helper functions for PyTorch
├── s1_enc.py                 # Stage 1: Feature encoding
├── s1.5_local_cls.py         # Stage 1.5: Local classifier training
├── s2_g.py                   # Stage 2: M-AR generator training
├── s2.5_sample.py            # Stage 2.5: Synthetic data sampling
├── s3_global_training.py     # Stage 3: Global classifier training
├── test.py                   # Testing script
├── models/                   # Model definitions
│   ├── encoders/            # Pretrained encoders (DINO, CLIP, etc.)
│   ├── mar_generator.py     # Mixture-of-Autoregressive generator
│   └── global_classifier.py # Global classifier model
├── dataset/                  # Raw dataset directory
├── dataset_preprocessed/     # Preprocessed dataset directory
├── feature/                  # Extracted features
├── weights/                  # Model weights
└── logs/                     # Training logs
```

## Dataset Preparation

### Dataset Directory Structure

The framework expects datasets to be organized in a specific hierarchical structure. The dataset should be placed in the `dataset_preprocessed/` directory.

#### General Format

```
/path/to/dataset_preprocessed/${DATASET_NAME}
│
├── ${SITE_ID}_train/                  <-- Training set for a specific Site/Client
│   ├── ${CLASS_ID}_${CLASS_NAME}/     <-- Class folder (Label 0)
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   ├── ${CLASS_ID}_${CLASS_NAME}/     <-- Class folder (Label 1)
│   │   └── ...
│   └── ... (More classes)
│
├── ${SITE_ID}_val/                    <-- Validation set for the same Site/Client
│   ├── ${CLASS_ID}_${CLASS_NAME}/
│   │   └── ...
│   └── ...
│
├── ${SITE_ID}_test/                   <-- Test set for the same Site/Client
│   ├── ${CLASS_ID}_${CLASS_NAME}/
│   │   └── ...
│   └── ...
│
├── ${NEXT_SITE_ID}_train/             <-- Next Site/Client
│   └── ...
└── ...
```

#### Format Specifications

- **${DATASET_NAME}**: Name of the dataset (e.g., `OfficeHome`, `PACS`, `ChestDR`, `DR`), which you should register them in the `config.py` file.
- **${SITE_ID}**: Site/Client identifier (e.g., `Client0`, `Client1`, `Client2`, or domain names like `Art`, `Clipart`), which you should register them in the `config.py` file.
- **${CLASS_ID}**: Zero-based numeric class index (e.g., `0`, `1`, `2`).
- **${CLASS_NAME}**: Descriptive class name (e.g., `others`, `nodule`, `Alarm_Clock`), which can be parsed automatically.
- **image_001.png**: Image files in common formats (`.png`, `.jpg`, `.jpeg`, etc.), which can be any name you like.

#### Example

For a binary classification dataset (e.g., ChestDR with 2 classes and 3 clients):

```
dataset_preprocessed/
└── ChestDR/
    ├── Client0_train/
    │   ├── 0_others/
    │   │   ├── img_001.png
    │   │   ├── img_002.png
    │   │   └── ...
    │   └── 1_nodule/
    │       ├── img_001.png
    │       ├── img_002.png
    │       └── ...
    ├── Client0_val/
    │   └── ... (same structure)
    ├── Client0_test/
    │   └── ... (same structure)
    ├── Client1_train/
    │   └── ... (same structure)
    ├── Client1_val/
    │   └── ... (same structure)
    ├── Client1_test/
    │   └── ... (same structure)
    ├── Client2_train/
    │   └── ... (same structure)
    ├── Client2_val/
    │   └── ... (same structure)
    └── Client2_test/
        └── ... (same structure)
```

#### Important Notes

1. **Class Index Prefix**: Each class folder must start with a numeric index followed by an underscore (`_`), e.g., `0_class_name`, `1_class_name`, etc.
2. **Consistent Class Order**: All clients must have the same class indices and names in the same order.
3. **Splits**: The framework supports `train`, `val`, and `test` splits. At minimum, `train` split is required.
4. **Multiple Clients**: The number of clients can vary depending on the dataset and federated learning setup.
5. **Image Formats**: Supported image formats include `.png`, `.jpg`, `.jpeg`, `.bmp`, etc.
6. **Resolution**: Images are automatically resized to the resolution specified in the configuration (e.g., 224×224, 448×448).

## Configuration

The `config.py` file contains all configuration parameters:

### Dataset Configuration

```python
DATASET = 'OfficeHome'  # Select dataset

DATASETS = {
    'OfficeHome': {
        'num_classes': 65,
        'resolution': 224,
        'clients': ['Art', 'Clipart', 'Product', 'Real_World']
    },
    # ... more datasets
}
```

### Encoder Configuration

```python
ENCODER = {
    'type': 'dino_base',  # Options: 'dino_base', 'clip_b16', 'bmclip_b16', etc.
}
```

## Usage

### Stage 1: Feature Encoding

Extract features from images using pretrained encoders:

```bash
python s1_enc.py --gpus 0
```

### Stage 1.5: Local Classifier Training

Train local classifiers for each client:

```bash
python s1.5_local_cls.py --gpu 0
```

### Stage 2: M-AR Generator Training

Train the Mixture-of-Autoregressive generator:

```bash
python s2_g.py --gpus 0
```

### Stage 2.5: Synthetic Data Sampling

Generate synthetic features:

```bash
python s2.5_sample.py --gpus 0
```

### Stage 3: Global Classifier Training

Train the global classifier with knowledge distillation:

```bash
python s3_global_training.py --gpu 0
```

### Testing

Evaluate the trained model:

```bash
python test.py --gpu 0
```

## Training Pipeline

The FALCON framework follows a multi-stage training pipeline:

1. **Stage 1 - Feature Encoding**: Extract features from raw images using pretrained vision encoders (DINO, CLIP, etc.)
2. **Stage 1.5 - Local Classifier Training**: Train local classifiers on each client's real features
3. **Stage 2 - M-AR Generator Training**: Train a Mixture-of-Autoregressive generator to model feature distributions
4. **Stage 2.5 - Synthetic Data Sampling**: Generate synthetic features to augment training data
5. **Stage 3 - Global Classifier Training**: Train a global classifier using both real and synthetic features with knowledge distillation

## Citation

If you use the FALCON framework in your research, please consider citing our work.

## License

Please refer to the LICENSE file in the project for details.
