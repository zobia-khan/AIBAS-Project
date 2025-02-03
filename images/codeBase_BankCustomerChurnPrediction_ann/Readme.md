# Bank Customer Churn Prediction
## Code Base Image ANN

---

## Overview

This repo is created as part of the course **"M. Grum: Advanced AI-based Application Systems"** by the **"Junior Chair for Business InformaGon Science, esp. AI-based ApplicaGon Systems"** at **"University of Potsdam"**

---

## Ownership

**Created by:** Unaiza Afridi & Zobia Khan

---

## Short Characterization of AI model

This project also utilizes an **Artificial Neural Network (ANN)** to predict customer churn in a banking environment. 

- **Preprocessing**: Loads and processes training and test datasets.
- **Model Architecture**:
  - **Input Layer**: 64 neurons (ReLU activation).
  - **Hidden Layer**: 32 neurons (ReLU activation) with dropout (0.3).
  - **Output Layer**: 1 neuron (sigmoid activation) for binary classification.
  - **Loss Function**: `binary_crossentropy`
  - **Optimizer**: `adam`
- **Training**:
  - Runs for **20 epochs** with a batch size of **32**.
  - Tracks loss and accuracy metrics.
- **Performance Tracking**:
  - Saves **training history plots** (accuracy & loss curves).
  - Generates **diagnostic plots** (Residual Analysis, Q-Q Plot, etc.).
  - Stores **performance metrics** in a text file.
- **Model Output**:
  - Trained model is saved as `ann_model.h5`.
  - All plots and metrics are stored in `../../learningBase/`.

---

## Commitment to License

According to the course requirement, this project is licensed under **"AGPL-3.0 license"**

