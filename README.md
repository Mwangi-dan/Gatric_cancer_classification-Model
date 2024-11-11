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
