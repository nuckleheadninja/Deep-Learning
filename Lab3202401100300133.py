# ============================================================
# Custom CNN on CIFAR-10 with Data Augmentation & Dropout
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ──────────────────────────────────────────────────────────────
# 1. CIFAR-10 Class Names
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# ──────────────────────────────────────────────────────────────
# 2. Load & Preprocess Data
# ──────────────────────────────────────────────────────────────
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

print(f"Training samples : {x_train.shape[0]}")
print(f"Test samples     : {x_test.shape[0]}")
print(f"Image shape      : {x_train.shape[1:]}")

# ──────────────────────────────────────────────────────────────
# 3. Data Augmentation
# ──────────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=36,          # random rotation up to 15°
    width_shift_range=0.1,      # horizontal shift up to 10%
    height_shift_range=0.1,     # vertical shift up to 10%
    horizontal_flip=True,       # random horizontal flip
    zoom_range=0.1,             # random zoom up to 10%
    fill_mode="nearest"
)
datagen.fit(x_train)

# ──────────────────────────────────────────────────────────────
# 4. Build the Custom CNN
# ──────────────────────────────────────────────────────────────
def build_model():
    model = models.Sequential(name="Custom_CIFAR10_CNN")

    # ── Conv Block 1 ──────────────────────────────────────────
    model.add(layers.Conv2D(32, (3, 3), padding="same",
                            input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Conv Block 2 ──────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Classifier Head ──────────────────────────────────────
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))

    return model


model = build_model()
model.summary()

# ──────────────────────────────────────────────────────────────
# 5. Compile the Model
# ──────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ──────────────────────────────────────────────────────────────
# 6. Callbacks
# ──────────────────────────────────────────────────────────────
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=2
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ──────────────────────────────────────────────────────────────
# 7. Train the Model
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50

print("\n Training started...\n")
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test_cat),
    callbacks=[lr_scheduler, early_stop],
    verbose=1
)

# ──────────────────────────────────────────────────────────────
# 8. Evaluate the Model
# ──────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n{'='*50}")
print(f" Test Accuracy : {test_acc * 100:.2f}%")
print(f" Test Loss     : {test_loss:.4f}")
print(f"{'='*50}\n")

# ──────────────────────────────────────────────────────────────
# 9. Plot Training Curves
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history["accuracy"],    label="Train Accuracy", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",  linewidth=2)
axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────────────────────
# 10. Confusion Matrix
# ──────────────────────────────────────────────────────────────
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = y_test.flatten()

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────────────────────
# 11. Classification Report
# ──────────────────────────────────────────────────────────────
print("\n Classification Report:\n")
print(classification_report(y_true_classes, y_pred_classes,
                            target_names=CLASS_NAMES))

# ──────────────────────────────────────────────────────────────
# 12. Sample Predictions
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
indices = np.random.choice(len(x_test), 15, replace=False)

for i, ax in enumerate(axes.flat):
    idx = indices[i]
    ax.imshow(x_test[idx])
    true_label = CLASS_NAMES[y_true_classes[idx]]
    pred_label = CLASS_NAMES[y_pred_classes[idx]]
    color = "green" if true_label == pred_label else "red"
    ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                 fontsize=9, color=color, fontweight="bold")
    ax.axis("off")

plt.suptitle("Sample Predictions (Green = Correct, Red = Wrong)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────────────────────
# 13. Save the Model
# ──────────────────────────────────────────────────────────────
model.save("cifar10_custom_cnn.h5")
print("\n Model saved as 'cifar10_custom_cnn.h5'")
