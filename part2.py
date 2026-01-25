# ============================================================
# PART 2 — NLP: EMBEDDINGS + RNN FOR TEXT CLASSIFICATION
# ============================================================
# This notebook/script strictly follows the rubric for Part 2:
# 2.1 Text Preparation, Tokenization, and Padding
# 2.2 Baseline Text Model (Embedding + Pooling)
# 2.3 RNN Model (LSTM)
# 2.4 Comparison + Why Transformers Help

# =====================
# Imports
# =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

# =====================
# Global constants
# =====================
MAX_VOCAB = 10000
MAX_LEN = 50
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2.1 Text Preparation, Tokenization, and Padding (10 Marks)
# ============================================================

print("\n=== 2.1 Text Preparation, Tokenization, Padding ===")

# Load data
df = pd.read_csv("C:/Tej/BITSoM/Projects/Assignment/DL Assignment/Dataset/text.csv")

# Inspect labels
print("Label distribution:\n", df['label'].value_counts())

# Minimal cleaning
texts = df['text'].astype(str).str.lower().str.strip().values
labels = df['label'].values

# Train / Val / Test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

# Tokenizer (fit on training text only)
# MAX_VOCAB = 10000
# MAX_LEN = 50

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
train_seq = tokenizer.texts_to_sequences(X_train)
val_seq = tokenizer.texts_to_sequences(X_val)
test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
X_train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post')
X_val_pad = pad_sequences(val_seq, maxlen=MAX_LEN, padding='post')
X_test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post')

# Show vocabulary size
vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
print("Final vocabulary size:", vocab_size)

# Show one worked example
sample_text = X_train[0]
sample_seq = tokenizer.texts_to_sequences([sample_text])[0]
sample_pad = pad_sequences([sample_seq], maxlen=MAX_LEN, padding='post')[0]

print("\nExample:")
print("Raw text:", sample_text)
print("Token IDs:", sample_seq)
print("Padded sequence:", sample_pad)

# Justification:
# A maximum sequence length of 50 was chosen as it captures most sentences
# while keeping computation efficient. Longer sequences would increase
# training time with diminishing returns for this dataset.

# ============================================================
# 2.2 Baseline Text Model (Embedding + Pooling) (10 Marks)
# ============================================================

print("\n=== 2.2 Baseline Model: Embedding + Pooling ===")

baseline_model = Sequential([
    Embedding(vocab_size, 64, input_length=MAX_LEN),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

baseline_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

baseline_model.summary()

start = time.time()
history_base = baseline_model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    verbose=1
)
base_time = time.time() - start

# Evaluation
y_test_pred = (baseline_model.predict(X_test_pad) > 0.5).astype(int)

base_acc = accuracy_score(y_test, y_test_pred)
base_f1 = f1_score(y_test, y_test_pred)
base_cm = confusion_matrix(y_test, y_test_pred)

print("Baseline Accuracy:", base_acc)
print("Baseline F1-score:", base_f1)
print("Baseline Confusion Matrix:\n", base_cm)

# ============================================================
# 2.3 RNN Model (LSTM) (10 Marks)
# ============================================================

print("\n=== 2.3 RNN Model: LSTM ===")

rnn_model = Sequential([
    Embedding(vocab_size, 64, input_length=MAX_LEN),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

rnn_model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

start = time.time()
history_rnn = rnn_model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    callbacks=[early_stop],
    verbose=1
)
rnn_time = time.time() - start

# Evaluation
y_test_pred_rnn = (rnn_model.predict(X_test_pad) > 0.5).astype(int)

rnn_acc = accuracy_score(y_test, y_test_pred_rnn)
rnn_f1 = f1_score(y_test, y_test_pred_rnn)
rnn_cm = confusion_matrix(y_test, y_test_pred_rnn)

print("RNN Accuracy:", rnn_acc)
print("RNN F1-score:", rnn_f1)
print("RNN Confusion Matrix:\n", rnn_cm)

# Explanation:
# LSTM was chosen over SimpleRNN because it can better capture long-range
# dependencies and mitigate the vanishing gradient problem, making it more
# suitable for sequential text data.

# ============================================================
# 2.4 Comparison + Why Transformers Help (4 Marks)
# ============================================================

print("\n=== 2.4 Comparison ===")

comparison = pd.DataFrame({
    'Model': ['Embedding + Pooling', 'LSTM'],
    'Accuracy': [base_acc, rnn_acc],
    'F1-score': [base_f1, rnn_f1],
    'Training Time (s)': [base_time, rnn_time]
})

print(comparison)

# Explanation:
# RNNs process text sequentially, which makes them slow and prone to
# vanishing gradients on long sequences. Transformers address this
# limitation using self-attention, allowing them to model global
# dependencies in parallel. This results in better scalability,
# improved performance on long texts, and faster training on modern
# hardware compared to RNN-based models.

print("\nPart 2 completed successfully.")