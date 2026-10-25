
# ============================================================
# PROJECT: IoT-Based Early Crop Disease Detection System
# MODEL: MobileNetV2 with Transfer Learning
# FILE: train_model.py
# ============================================================

import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------------------------------------------------
# STEP 1: CONFIGURATION
# ------------------------------------------------------------

# Replace this example with the path from inspect_dataset.py
DATASET_PATH = r"C:\Users\cw\Desktop\Disease detection\color"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

MODEL_PATH = "crop_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# ------------------------------------------------------------
# STEP 2: CHECK THE DATASET PATH
# ------------------------------------------------------------

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        "Dataset folder not found. Check DATASET_PATH."
    )

print("Dataset found:", DATASET_PATH)

# ------------------------------------------------------------
# STEP 3: LOAD TRAINING AND VALIDATION IMAGES
# ------------------------------------------------------------

# The dataset must contain one subfolder per class.
# 80% of images will be used for training and 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# Save the class names in the same order used by the dataset.
class_names = train_ds.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)
print("Class names:", class_names)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# ------------------------------------------------------------
# STEP 4: IMPROVE DATA LOADING PERFORMANCE
# ------------------------------------------------------------

# Prefetch allows TensorFlow to prepare the next batch
# while the current batch is being processed.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------
# STEP 5: DATA AUGMENTATION
# ------------------------------------------------------------

# These transformations create slightly varied training
# examples to help the model generalize.
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
])

# ------------------------------------------------------------
# STEP 6: LOAD THE PRETRAINED MOBILENETV2 MODEL
# ------------------------------------------------------------

# ImageNet weights provide useful visual features learned
# from a large image dataset.
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers during the first training stage.
base_model.trainable = False

# ------------------------------------------------------------
# STEP 7: BUILD OUR DISEASE CLASSIFIER
# ------------------------------------------------------------

inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

# Convert pixels into the range expected by MobileNetV2.
x = layers.Lambda(preprocess_input)(x)

# Extract features from the leaf image.
x = base_model(x, training=False)

# Reduce the feature maps to a feature vector.
x = layers.GlobalAveragePooling2D()(x)

# Reduce overfitting.
x = layers.Dropout(0.2)(x)

# Predict a probability for each disease class.
outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ------------------------------------------------------------
# STEP 8: COMPILE THE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------------
# STEP 9: SET TRAINING CALLBACKS
# ------------------------------------------------------------

callbacks = [
    # Stop training if validation loss stops improving.
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    # Save the best model based on validation loss.
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]

# ------------------------------------------------------------
# STEP 10: TRAIN THE CLASSIFIER
# ------------------------------------------------------------

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the final model state as well.
model.save(MODEL_PATH)

print("\nTraining finished!")
print("Saved model:", MODEL_PATH)
print("Saved class names:", CLASS_NAMES_PATH)
print("Classes:", num_classes)
