# BITSoM-DL_Assignment
BITSoM Deep Learning assignment
# Deep Learning Foundations — Assignment Solutions

This repository contains end-to-end implementations of Deep Learning models for **tabular data, natural language processing, and computer vision**, completed as part of the *Deep Learning Foundations* coursework.

The solutions are designed to be:
- Fully reproducible
- Clearly structured
- Aligned with academic evaluation rubrics
- Explained with theory-first reasoning

---

## 📂 Repository Structure
<pre>
├── notebooks/
│ ├── DL_Part1_Tabular_Regression.ipynb
│ ├── DL_Part2_Text_Classification.ipynb
│ └── DL_Part3_Image_Classification.ipynb
│
├── Dataset/
│ ├── tabular.csv
│ ├── text.csv
│ └── images/
│ ├── class_0/
│ └── class_1/
│
├── outputs/
│ ├── datasetA_loss_curve.png
│ ├── keras_ffnn_loss.png
│ ├── keras_parity_plot.png
│ ├── part3_training_loss.png
│ └── part3_misclassified_images.png
│
├── requirements.txt
└── README.md

</pre>

## 🧠 Assignment Overview

The assignment is divided into **three parts**, each addressing a different data modality and learning objective.

| Part | Domain | Task Type | Model Used |
|----|----|----|----|
| Part 1 | Tabular Data | Regression | Feedforward Neural Network |
| Part 2 | Text (NLP) | Binary Classification | Embeddings + Pooling, LSTM |
| Part 3 | Images (CV) | Image Classification | Convolutional Neural Network |

---

## 🔹 Part 1 — Tabular Regression

### Objective
Predict a continuous target variable from structured customer data.

### Key Steps
- Missing value handling (median/mode imputation)
- One-hot encoding of categorical variables
- Feature scaling using StandardScaler
- Feedforward Neural Network (FFNN) implementation
- Model training using PyTorch
- Comparison using a Keras-based FFNN
- Evaluation using RMSE and MAE

### Models Used
- **PyTorch FFNN** (manual training loop)
- **Keras FFNN** (with EarlyStopping)

### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)


## 🔹 Part 2 — NLP Text Classification

### Objective
Classify text data (e.g., customer feedback) into binary categories.

### Key Steps
- Text preprocessing (lowercasing, tokenization)
- Vocabulary construction using Keras Tokenizer
- Padding for fixed-length sequences
- Word Embedding layer for semantic representation
- Model comparison:
  - Embedding + Global Average Pooling (baseline)
  - LSTM-based RNN (sequence-aware model)

### Models Used
- Embedding + Pooling
- LSTM (Recurrent Neural Network)

### Evaluation Metrics
- Accuracy
- F1-score
- Confusion Matrix

## 🔹 Part 3 — Computer Vision (CNN)

### Objective
Classify images into two categories using a Convolutional Neural Network.

### Key Steps
- Image loading using `image_dataset_from_directory`
- Normalization using Rescaling layer
- Visual sanity checks
- CNN architecture design
- Model training with EarlyStopping
- Error analysis using confusion matrix and misclassified samples

### Model Architecture
- Convolutional layers for feature extraction
- MaxPooling for spatial downsampling
- Fully connected layers for classification

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

## 🛠 Technologies Used

- **Python**
- **PyTorch**
- **TensorFlow / Keras**
- **Scikit-learn**
- **NumPy / Pandas**
- **Matplotlib**


## ▶️ How to Run

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>

2. Install dependencies:
pip install -r requirements.txt

3. Open notebooks:
jupyter notebook

## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py

4. Run notebooks in order:
DL_Part1_Tabular_Regression.ipynb
DL_Part2_Text_Classification.ipynb
DL_Part3_Image_Classification.ipynb

***Notes:***

All notebooks are executed top-to-bottom.
Random seeds are fixed for reproducibility.
Plots are saved to the outputs/ directory.
Code is commented for academic clarity and viva readiness.

Author

Tej Yadav
Deep Learning & Business Analytics
(BITSoM Coursework)

License

This repository is intended for academic and educational purposes only.






