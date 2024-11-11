# Gastric Cancer Detection from Endoscopic Images

## Overview

Gastric cancer is a leading cause of cancer-related deaths worldwide, and early detection is crucial for improving patient outcomes. This project explores the application of machine learning to aid in the early detection of gastric cancer through endoscopic image analysis. Three models are implemented in this repository: a **CNN-based model using ResNet50**, an **SVM model with pre-extracted ResNet50 features**, and a **Random Forest model trained on hand-crafted image features**. These models aim to assist medical professionals by automating and enhancing the detection process.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [References](#references)

---

## Project Structure

```plaintext
├── data/                            # Directory containing endoscopic images from the Kvasir dataset
├── models/                          # Saved model files (e.g., best_model.h5)
├── notebooks/                       # Jupyter/Colab notebooks for training and testing models
├── README.md                        # Project documentation
└── src/
    ├── cnn_model.py                 # Code for CNN-based model
    ├── svm_model.py                 # Code for SVM model with ResNet50 feature extraction
    ├── rf_model.py                  # Code for Random Forest with hand-crafted features
    └── utils.py                     # Helper functions (data loading, feature extraction, etc.)
```


## Dataset

This project uses the **Kvasir dataset** for training and evaluation. The dataset includes various endoscopic images of the gastrointestinal tract, including images of normal and cancerous tissues.

**Note**: The Kvasir dataset should be organized in `data/` with subfolders for each class label (e.g., `normal`, `cancerous`).

- **Download**: The dataset is available from [Kvasir Dataset](https://datasets.simula.no/kvasir/).
- **Structure**:
  - Place images in subdirectories based on class labels (e.g., `data/normal/` and `data/cancerous/`).



## Models

This project implements three models with different machine learning techniques:

### 1. CNN-Based Model (ResNet50)

- **Approach**: Uses transfer learning with ResNet50, a pre-trained CNN architecture, to classify endoscopic images.
- **Purpose**: Leverages high-level image features for accurate binary classification of normal vs. cancerous tissue.
- **Code**: `src/cnn_model.py`

### 2. SVM Model with Pre-extracted ResNet50 Features

- **Approach**: ResNet50 is used as a feature extractor to convert images into embeddings, which are then classified with an SVM.
- **Purpose**: Provides a simpler ML model with CNN-derived features for faster training and better interpretability.
- **Code**: `src/svm_model.py`

### 3. Random Forest Model with Hand-Crafted Image Features

- **Approach**: Uses traditional image processing to extract color histograms and texture features (GLCM) from images, which are then classified using a Random Forest model.
- **Purpose**: An interpretable approach that combines hand-crafted features with a traditional ML algorithm.
- **Code**: `src/rf_model.py`


## Installation

To set up the environment and dependencies:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gastric-cancer-detection.git
   cd gastric-cancer-detection