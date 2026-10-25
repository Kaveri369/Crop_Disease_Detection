

# ============================================================
# CROP DISEASE DETECTION - MODEL EVALUATION
# Accuracy, Precision, Recall, F1-score and Confusion Matrix
# ============================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# 1. CONFIGURE FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR /  "color"
MODEL_PATH = BASE_DIR / "crop_disease_model.keras"
GRAPHS_PATH = BASE_DIR / "graphs"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Create the output directory if it does not exist.
GRAPHS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False,
    compile=False
)

print("Model loaded successfully!")


# ============================================================
# 3. LOAD THE VALIDATION DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_ds.class_names
number_of_classes = len(class_names)

print("Number of classes:", number_of_classes)


# ============================================================
# 4. COLLECT ACTUAL LABELS AND MODEL PREDICTIONS
# ============================================================

# Extract the true class index from every validation image.
actual_labels = np.concatenate([
    labels.numpy() for images, labels in validation_ds
])

# Predict the class scores for all validation images.
predicted_probabilities = model.predict(
    validation_ds,
    verbose=1
)

# Select the class with the highest predicted score.
predicted_labels = np.argmax(
    predicted_probabilities,
    axis=1
)

print("Total validation images:", len(actual_labels))


# ============================================================
# 5. CALCULATE OVERALL EVALUATION METRICS
# ============================================================

# Accuracy: proportion of correctly classified images.
accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

# Macro averages give equal weight to each class.
macro_precision = precision_score(
    actual_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    actual_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    actual_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
    average="macro",
    zero_division=0
)

# Weighted averages account for the number of true examples per class.
weighted_precision = precision_score(
    actual_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    actual_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    actual_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

# Store metrics as percentages for easier interpretation.
overall_metrics = {
    "accuracy": round(float(accuracy * 100), 4),
    "macro_precision": round(float(macro_precision * 100), 4),
    "macro_recall": round(float(macro_recall * 100), 4),
    "macro_f1_score": round(float(macro_f1 * 100), 4),
    "weighted_precision": round(float(weighted_precision * 100), 4),
    "weighted_recall": round(float(weighted_recall * 100), 4),
    "weighted_f1_score": round(float(weighted_f1 * 100), 4),
    "validation_images": int(len(actual_labels)),
    "number_of_classes": int(number_of_classes)
}

print("\n========== OVERALL MODEL METRICS ==========")

for metric_name, metric_value in overall_metrics.items():
    if isinstance(metric_value, float):
        print(f"{metric_name}: {metric_value:.2f}%")
    else:
        print(f"{metric_name}: {metric_value}")


# ============================================================
# 6. GENERATE THE CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
    actual_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
    target_names=class_names,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========\n")
print(report_text)

# Save a readable text report.
with open(
    GRAPHS_PATH / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report_text)

# Convert the classification report into a dictionary.
report_dict = classification_report(
    actual_labels,
    predicted_labels,
    labels=list(range(number_of_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

# Save overall metrics as JSON.
with open(
    GRAPHS_PATH / "overall_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(overall_metrics, file, indent=4)

# Save detailed metrics for every class as CSV.
class_metrics = pd.DataFrame(
    [
        {
            "Class": class_name,
            "Precision": report_dict[class_name]["precision"],
            "Recall": report_dict[class_name]["recall"],
            "F1_Score": report_dict[class_name]["f1-score"],
            "Support": int(report_dict[class_name]["support"])
        }
        for class_name in class_names
    ]
)

class_metrics.to_csv(
    GRAPHS_PATH / "per_class_metrics.csv",
    index=False
)


# ============================================================
# 7. GRAPH 1: OVERALL METRICS COMPARISON
# ============================================================

metric_names = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1-score"
]

metric_values = [
    accuracy * 100,
    macro_precision * 100,
    macro_recall * 100,
    macro_f1 * 100
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    metric_names,
    metric_values
)

ax.set_title("Overall Crop Disease Model Performance")
ax.set_ylabel("Score (%)")
ax.set_xlabel("Evaluation Metric")
ax.set_ylim(0, 105)

ax.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3
)

ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(
    GRAPHS_PATH / "overall_metrics.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved: graphs/overall_metrics.png")


# ============================================================
# 8. GRAPH 2: PER-CLASS PRECISION, RECALL AND F1-SCORE
# ============================================================

# Display all classes in three aligned horizontal bar charts.
# Horizontal charts are easier to read with 38 class names.

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(21, max(12, number_of_classes * 0.38)),
    sharey=True
)

class_positions = np.arange(number_of_classes)

metrics_to_plot = [
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1_Score", "F1-score")
]

for ax, (column, title) in zip(axes, metrics_to_plot):

    values = class_metrics[column].to_numpy() * 100

    ax.barh(
        class_positions,
        values
    )

    ax.set_title(f"Per-Class {title}")
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax.set_yticks(class_positions)
    ax.set_yticklabels(class_names, fontsize=7)

# Reverse the vertical order so the first class appears at the top.
axes[0].invert_yaxis()

fig.suptitle(
    "Precision, Recall and F1-score for Each Crop Disease Class",
    fontsize=15
)

plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(
    GRAPHS_PATH / "per_class_metrics.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved: graphs/per_class_metrics.png")


# ============================================================
# 9. GRAPH 3: PRECISION, RECALL AND F1-SCORE HEATMAP
# ============================================================

heatmap_values = (
    class_metrics[["Precision", "Recall", "F1_Score"]]
    .to_numpy() * 100
)

fig, ax = plt.subplots(
    figsize=(9, max(12, number_of_classes * 0.35))
)

image = ax.imshow(
    heatmap_values,
    aspect="auto",
    vmin=0,
    vmax=100,
    cmap="YlGnBu"
)

ax.set_title("Per-Class Classification Metrics Heatmap")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Precision", "Recall", "F1-score"])

ax.set_yticks(np.arange(number_of_classes))
ax.set_yticklabels(class_names, fontsize=7)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Score (%)")

plt.tight_layout()

plt.savefig(
    GRAPHS_PATH / "metrics_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved: graphs/metrics_heatmap.png")


# ============================================================
# 10. GRAPH 4: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(number_of_classes))
)

fig, ax = plt.subplots(figsize=(22, 22))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation="vertical",
    include_values=False,
    cmap="Blues",
    colorbar=False
)

ax.set_title("Crop Disease Classification Confusion Matrix")

plt.tight_layout()

plt.savefig(
    GRAPHS_PATH / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved: graphs/confusion_matrix.png")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n========== EVALUATION COMPLETED ==========")
print("All evaluation files saved in:", GRAPHS_PATH.resolve())

print("\nGenerated files:")
print("1. overall_metrics.png")
print("2. per_class_metrics.png")
print("3. metrics_heatmap.png")
print("4. confusion_matrix.png")
print("5. overall_metrics.json")
print("6. per_class_metrics.csv")
print("7. classification_report.txt")
