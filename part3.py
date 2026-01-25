# Part 3 — Computer Vision: CNN Image Classifier

# This section implements an end-to-end CNN-based image classification pipeline using Keras, strictly following the rubric.


## 3.1 Loading, Normalization, and Visual Sanity Checks (10 Marks)
import os
print("Saving files to:", os.getcwd())

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42

OUTPUT_DIR = "C:/Tej/BITSoM/Projects/Assignment/DL Assignment/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_dir = "C:/Tej/BITSoM/Projects/Assignment/DL Assignment/Dataset/images"

train_ds = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Class names:", class_names)
print("Number of classes:", num_classes)

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Visual sanity check
plt.figure(figsize=(6,6))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.savefig("part3_sample_images.png")
plt.close()


## 3.2 Build and Train a CNN (14 Marks)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=IMAGE_SIZE + (3,)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop]
)

# Plot training curves
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("part3_training_loss.png")
plt.close()



## 3.3 Evaluation, Confusion Matrix, and Misclassification Review (10 Marks)

# Prepare test set
all_images = []
all_labels = []

for images, labels in val_ds:
    all_images.append(images)
    all_labels.append(labels)

X_test = tf.concat(all_images, axis=0)
y_test = tf.concat(all_labels, axis=0)

pred_probs = model.predict(X_test)
y_pred = np.argmax(pred_probs, axis=1)

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Show misclassified images
mis_idx = np.where(y_pred != y_test.numpy())[0]

plt.figure(figsize=(8,4))
for i, idx in enumerate(mis_idx[:5]):
    ax = plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx])
    plt.title(f"T:{class_names[y_test[idx]]}\nP:{class_names[y_pred[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("part3_misclassified_images.png")
plt.close()


### Misclassification Analysis (5–6 sentences)


print("**Part 3 completed successfully.**")
