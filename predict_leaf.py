
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# 1. Define file paths
# --------------------------------------------------

MODEL_PATH = Path("crop_disease_model.keras")
CLASS_NAMES_PATH = Path("class_names.json")
IMAGE_PATH = Path("test_images") / "test_leaf.jpg"

IMAGE_SIZE = (160, 160)

# --------------------------------------------------
# 2. Check that required files exist
# --------------------------------------------------

for path in [MODEL_PATH, CLASS_NAMES_PATH, IMAGE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.resolve()}")

# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False,
    compile=False
)

# Load class labels in the original order
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

# --------------------------------------------------
# 4. Load and prepare the leaf image
# --------------------------------------------------

image = load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
image_array = img_to_array(image)

# Match the MobileNetV2 preprocessing used during training
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)

# --------------------------------------------------
# 5. Predict the class
# --------------------------------------------------

predictions = model.predict(image_array, verbose=0)[0]

predicted_index = int(np.argmax(predictions))
predicted_class = class_names[predicted_index]
confidence = float(predictions[predicted_index]) * 100

print("\n========== CROP DISEASE PREDICTION ==========")
print("Predicted class:", predicted_class)
print(f"Model confidence: {confidence:.2f}%")

# Display the image and prediction
plt.figure(figsize=(6, 5))
plt.imshow(image)
plt.title(f"{predicted_class}\nConfidence: {confidence:.2f}%")
plt.axis("off")
plt.tight_layout()
plt.show()
