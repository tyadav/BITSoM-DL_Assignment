# Deep Learning Foundations Assignment
# ============================================================
#   EXPLANATION & THEORY (FOR GRADERS)
# ============================================================
# 1. Neuron computation:
#    Each layer computes: y = activation(Wx + b)
#    - W: learned weights
#    - b: learned bias
#    - activation: ReLU for hidden layers to introduce non-linearity
#
# 2. Loss minimization:
#    - Regression uses Mean Squared Error (MSE)
#    - Classification uses Cross Entropy Loss
#    Training uses gradient descent via backpropagation (Adam optimizer)
#
# 3. Output activations:
#    - Regression: Linear output (no activation)
#    - Classification: Raw logits (softmax handled internally by CrossEntropyLoss)
#
# 4. Evaluation:
#    - Regression: RMSE + train/val loss curves
#    - Classification: Accuracy, confusion matrix, classification report
#
# 5. Overfitting / Underfitting detection:
#    - Overfitting: training loss ↓ while validation loss ↑
#    - Underfitting: both losses high and not improving
# ============================================================

# Deep Learning Foundations Assignment
# End-to-end, runnable baseline solutions for Tabular, NLP, and CV tasks
# Author: Tej Yadav

# ============================================================
# 1.1 Single Neuron Forward Pass (6 Marks)
# ============================================================

import numpy as np

def single_neuron_forward(x, w, b):
    z = np.dot(w, x) + b
    relu = np.maximum(0, z)
    sigmoid = 1 / (1 + np.exp(-z))
    return z, relu, sigmoid

# Worked example
x = np.array([1.5, -2.0, 3.0])
w = np.array([0.4, -0.6, 0.2])
b = 0.5

z, relu_z, sigmoid_z = single_neuron_forward(x, w, b)

print("\n=== 1.1 Single Neuron Forward Pass ===")
print("z =", z)
print("ReLU(z) =", relu_z)
print("Sigmoid(z) =", sigmoid_z)

# Explanation:
# Activation functions introduce non-linearity, enabling neural networks
# to model complex relationships. ReLU is commonly used in hidden layers
# due to its computational efficiency and reduced vanishing-gradient issue.
# Sigmoid is typically used in output layers for binary classification
# where probabilistic interpretation is required.

# =====================
#  Common Setup
# =====================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ============================================================
# DATASET A — TABULAR REGRESSION (REVISED & STABLE)
# ============================================================
print("\n=== Dataset A: Tabular Regression ===")

tabular_path = "C:/Tej/BITSoM/Projects/Assignment/DL Assignment/Dataset/tabular.csv"
df = pd.read_csv(tabular_path)

# -------------------------
# 1. Basic data checks
# -------------------------
print("Missing values per column:")
print(df.isna().sum())

# Drop rows with missing target (cannot train without target)
df = df.dropna(subset=["target"])

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"].values.reshape(-1, 1)

# -------------------------
# 2. Handle missing values
# -------------------------
# Fill numeric missing values with column mean
# Separate numeric & categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

# Median for numeric
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# Mode for categorical
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# One-hot encode categoricals (safe baseline)
X = pd.get_dummies(X, drop_first=True)
X = X.astype(np.float32)   # <<< CRITICAL LINE


# -------------------------
# 3. Train / Validation split
# -------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X.values, y, test_size=0.30, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

print("Shapes after preprocessing:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# -------------------------
# 4. Feature scaling
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Optional: clip extreme values to avoid exploding gradients
X_train = np.clip(X_train, -10, 10)
X_val = np.clip(X_val, -10, 10)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

# -------------------------
# 5. Model definition
# -------------------------
class TabularRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Linear output for regression
        )

    def forward(self, x):
        return self.net(x)

model_tab = TabularRegressor(X_train.shape[1]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_tab.parameters(), lr=1e-3)

# -------------------------
# 6. Training loop
# -------------------------
def train_regression(model, X_tr, y_tr, X_v, y_v, epochs=50):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_tr.to(DEVICE))
        loss = criterion(preds, y_tr.to(DEVICE))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_v.to(DEVICE))
            val_loss = criterion(val_preds, y_v.to(DEVICE))

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train MSE={loss.item():.4f}, val MSE={val_loss.item():.4f}")

    return train_losses, val_losses

train_losses, val_losses = train_regression(
    model_tab,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t
)

# -------------------------
# 7. Evaluation (RMSE)
# -------------------------
model_tab.eval()
with torch.no_grad():
    val_preds = model_tab(X_val_t.to(DEVICE)).cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE:", rmse)

# -------------------------
# 8. Loss curves
# -------------------------
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Tabular Regression Loss")
plt.legend()
plt.tight_layout()

plt.savefig("datasetA_loss_curve.png")
plt.close()

# ============================================================
# 1.3 FFNN for Regression using Keras (12 Marks)
# ============================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

# ---- FORCE NUMERIC DTYPE FOR KERAS ----
X_train = np.asarray(X_train, dtype=np.float32)
X_val   = np.asarray(X_val,   dtype=np.float32)
X_test  = np.asarray(X_test,  dtype=np.float32)

model_keras = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model_keras.compile(
    optimizer="adam",
    loss="mse"
)

print("\nKeras Model Summary:")
model_keras.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model_keras.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[early_stop],
    verbose=1
)

# Loss plot
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Keras FFNN Training vs Validation Loss")
plt.legend()
plt.savefig("keras_ffnn_loss.png")
plt.close()

# Test evaluation
y_test_pred = model_keras.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Keras Test MAE:", mae)
print("Keras Test RMSE:", rmse)

# Parity plot
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Parity Plot (Keras FFNN)")
plt.savefig("keras_parity_plot.png")
plt.close()

# ============================================================
# 1.4 Overfitting / Underfitting Diagnosis (4 Marks)
# ============================================================

# The training and validation loss curves decrease steadily and remain
# close to each other, indicating that the model is reasonably well-balanced.
# There is no strong evidence of overfitting, as validation loss does not
# diverge from training loss. Early stopping further prevents memorization.
# Minor residual error suggests limited model capacity.
# To improve performance, regularization techniques such as dropout or L2
# regularization could be added. Feature engineering or additional data
# collection could further enhance generalization.

# ============================================================
# DATASET B — NLP TEXT CLASSIFICATION
# ============================================================
print("\n=== Dataset B: NLP Text Classification ===")

text_path = "C:/Tej/BITSoM/Projects/Assignment/DL Assignment/Dataset/text.csv"
df_text = pd.read_csv(text_path)

texts = df_text["text"].astype(str).values
labels = df_text["label"].values

# Build vocabulary (simple baseline)
from collections import Counter

def tokenize(text):
    return text.lower().split()

counter = Counter()
for t in texts:
    counter.update(tokenize(t))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(5000):
    vocab[word] = len(vocab)

max_len = 50

def encode(text):
    tokens = tokenize(text)
    ids = [vocab.get(tok, 1) for tok in tokens][:max_len]
    return ids + [0] * (max_len - len(ids))

X_enc = np.array([encode(t) for t in texts])

X_train, X_val, y_train, y_val = train_test_split(
    X_enc, labels, test_size=0.2, random_state=SEED
)

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=32)

num_classes = len(np.unique(labels))

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)           # (B, T, D)
        pooled = emb.mean(dim=1)          # average pooling
        return self.fc(pooled)

model_text = TextClassifier(len(vocab), 100, num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_text.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model_text.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model_text(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    model_text.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model_text(xb.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    print(f"Epoch {epoch}: val accuracy={acc:.4f}")

print(classification_report(all_true, all_preds))

# ============================================================
# DATASET C — COMPUTER VISION IMAGE CLASSIFICATION
# ============================================================
print("\n=== Dataset C: Computer Vision ===")

# Expected directory structure:
# images/
#   class_0/
#   class_1/

from torchvision import datasets, transforms

image_root = "C:/Tej/BITSoM/Projects/Assignment/DL Assignment/Dataset/images"  # change if needed

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(image_root, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model_cv = SimpleCNN(len(full_dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cv.parameters(), lr=1e-3)

for epoch in range(10):
    model_cv.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model_cv(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    model_cv.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model_cv(xb.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    print(f"Epoch {epoch}: val accuracy={acc:.4f}")

print("CV Confusion Matrix:\n", confusion_matrix(all_true, all_preds))
print("\nAll baselines trained successfully.")
