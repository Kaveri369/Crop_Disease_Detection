

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)


# =========================================================
# 2. DISEASE INFORMATION AND PREVENTION SUGGESTIONS
# =========================================================

DISEASE_INFO = {
    "healthy": {
        "description": (
            "The model predicts that the leaf belongs to a "
            "healthy class. Continue monitoring the plant "
            "for changes in leaf colour, spots, or wilting."
        ),
        "symptoms": "No disease symptoms are indicated by this class.",
        "prevention": [
            "Use clean water and maintain suitable soil moisture.",
            "Provide appropriate sunlight and balanced nutrition.",
            "Inspect leaves regularly for unusual spots or yellowing.",
            "Use healthy seeds or planting material.",
        ],
    },

    "early blight": {
        "description": (
            "Early blight is a plant disease commonly associated "
            "with fungal infection. It often affects tomato and potato."
        ),
        "symptoms": (
            "Possible symptoms include brown spots, often with "
            "concentric rings, and yellowing around affected areas."
        ),
        "prevention": [
            "Avoid wetting foliage when watering; water near the soil.",
            "Remove badly affected leaves when appropriate.",
            "Clean tools after working with affected plants.",
            "Rotate crops and remove infected plant debris.",
            "Allow adequate spacing for airflow around plants.",
        ],
    },

    "late blight": {
        "description": (
            "Late blight is a potentially serious disease of "
            "potato and tomato, especially in cool, wet conditions."
        ),
        "symptoms": (
            "Possible signs include rapidly expanding dark or "
            "water-soaked leaf patches and pale growth near lesions "
            "under favourable conditions."
        ),
        "prevention": [
            "Use healthy planting material and suitable resistant varieties.",
            "Avoid prolonged leaf wetness and improve airflow.",
            "Inspect plants frequently during cool, wet weather.",
            "Remove infected material appropriately.",
            "Seek local agricultural advice promptly if suspected.",
        ],
    },

    "bacterial spot": {
        "description": (
            "Bacterial spot can affect crops such as tomato and pepper. "
            "It may spread through infected material and water splash."
        ),
        "symptoms": (
            "Small dark or water-soaked spots may appear on leaves "
            "and sometimes on fruit."
        ),
        "prevention": [
            "Use disease-free seeds and seedlings.",
            "Avoid handling plants when foliage is wet.",
            "Reduce splashing water and avoid overhead irrigation.",
            "Clean tools and remove infected crop debris.",
            "Ask a local agricultural expert to confirm the diagnosis.",
        ],
    },

    "leaf mold": {
        "description": (
            "Leaf mold commonly affects tomato foliage and is "
            "favoured by high humidity and poor ventilation."
        ),
        "symptoms": (
            "Yellow patches may appear on upper leaf surfaces, "
            "with olive or greyish growth underneath."
        ),
        "prevention": [
            "Improve ventilation and spacing between plants.",
            "Water near the soil instead of wetting leaves.",
            "Avoid excessive humidity where crops are grown under cover.",
            "Remove affected plant debris and monitor new growth.",
        ],
    },

    "septoria": {
        "description": (
            "Septoria leaf spot is a fungal disease commonly "
            "associated with tomato plants."
        ),
        "symptoms": (
            "Small circular spots with pale centres and dark borders "
            "may develop, often on lower leaves."
        ),
        "prevention": [
            "Avoid splashing soil onto leaves during watering.",
            "Remove affected lower leaves when appropriate.",
            "Use mulch to reduce soil splash.",
            "Rotate crops and remove infected debris.",
            "Keep tools clean and allow good airflow.",
        ],
    },

    "common rust": {
        "description": (
            "Common rust is a fungal disease that can affect corn "
            "and produce rust-coloured pustules on leaves."
        ),
        "symptoms": (
            "Small reddish-brown or orange-brown raised pustules "
            "may appear on leaf surfaces."
        ),
        "prevention": [
            "Choose locally recommended resistant corn varieties.",
            "Inspect plants regularly for rust-coloured pustules.",
            "Manage volunteer corn and weeds where appropriate.",
            "Seek local agricultural advice if disease increases.",
        ],
    },

    "gray leaf spot": {
        "description": (
            "Gray leaf spot is a corn disease favoured by prolonged "
            "leaf wetness and humid conditions."
        ),
        "symptoms": (
            "Elongated rectangular grey or brown lesions may develop "
            "between leaf veins."
        ),
        "prevention": [
            "Use resistant or tolerant varieties where available.",
            "Rotate crops when appropriate.",
            "Manage infected crop residues according to local guidance.",
            "Monitor fields during warm, humid weather.",
        ],
    },

    "northern leaf blight": {
        "description": (
            "Northern leaf blight is a fungal disease of corn."
        ),
        "symptoms": (
            "Long, cigar-shaped grey-green or tan lesions may "
            "appear on leaves."
        ),
        "prevention": [
            "Choose locally recommended resistant varieties.",
            "Rotate crops when appropriate.",
            "Manage infected crop residues.",
            "Monitor plants and seek local agricultural advice.",
        ],
    },

    "powdery mildew": {
        "description": (
            "Powdery mildew is a fungal disease that can affect "
            "many plant species."
        ),
        "symptoms": (
            "White or grey powder-like patches may appear on "
            "leaf surfaces."
        ),
        "prevention": [
            "Provide suitable spacing and airflow.",
            "Remove severely affected plant material where appropriate.",
            "Avoid excessive nitrogen fertilisation.",
            "Monitor new growth and identify the crop before treatment.",
        ],
    },

    "mosaic virus": {
        "description": (
            "Mosaic virus diseases can cause irregular leaf colouring "
            "and distorted plant growth."
        ),
        "symptoms": (
            "Possible symptoms include alternating light and dark "
            "green areas, mottling, or distorted leaves."
        ),
        "prevention": [
            "Use certified or healthy planting material.",
            "Remove confirmed infected plants when advised locally.",
            "Clean hands and tools between plants.",
            "Manage insect vectors according to local guidance.",
            "Avoid saving seeds from suspected infected plants.",
        ],
    },
}


GENERAL_PREVENTION = [
    "Use healthy seeds or planting material.",
    "Inspect plants regularly and remove weeds around the crop.",
    "Avoid overwatering and provide suitable drainage.",
    "Maintain appropriate spacing and airflow.",
    "Clean gardening tools and manage infected plant debris.",
    "Confirm the disease before choosing a chemical treatment.",
]


# =========================================================
# 3. LOAD THE MODEL
# =========================================================

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
)

@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


# =========================================================
# 4. FIND RELEVANT DISEASE INFORMATION
# =========================================================

def get_disease_info(class_name):
    label = class_name.lower().replace("_", " ")
    label = label.replace("(", "").replace(")", "")

    if "healthy" in label:
        return DISEASE_INFO["healthy"], True

    # Match specific disease names in the predicted class.
    # More specific entries should be checked before general ones.
    matches = [
        "northern leaf blight",
        "gray leaf spot",
        "bacterial spot",
        "late blight",
        "early blight",
        "powdery mildew",
        "common rust",
        "leaf mold",
        "septoria",
        "mosaic virus",
    ]

    for disease in matches:
        if disease in label:
            return DISEASE_INFO[disease], False

    # Some PlantVillage labels use names such as
    # "Apple___Apple_scab" or "Grape___Black_rot".
    # Display a safe general guide if no specific entry matches.
    return {
        "description": (
            f"The model classified the image as {label.replace('___', ' — ')}. "
            "The app does not yet contain a verified, disease-specific "
            "description for this class."
        ),
        "symptoms": (
            "Compare the visible symptoms with trusted crop-specific "
            "guidance. Image predictions alone cannot confirm a diagnosis."
        ),
        "prevention": GENERAL_PREVENTION,
    }, False


# =========================================================
# 5. DASHBOARD HEADER
# =========================================================

st.title("🌿 Crop Disease Detection and Prevention System")

st.write(
    "Upload a crop leaf image to view the predicted class, "
    "possible symptoms, and prevention suggestions."
)

st.info(
    "AI model: MobileNetV2 | Dataset: PlantVillage | "
    "This is an academic prototype."
)


# =========================================================
# 6. UPLOAD AND DISPLAY IMAGE
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        model, class_names = load_model_and_classes()

        resized_image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(resized_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # Preprocessing is already included in the saved model.
        predictions = model.predict(image_array, verbose=0)[0]

        if len(predictions) != len(class_names):
            st.error(
                "The number of model outputs does not match "
                "the number of saved class names."
            )
            st.stop()

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        disease_info, is_healthy = get_disease_info(predicted_class)

        left, right = st.columns([1, 1])

        with left:
            st.subheader("📷 Uploaded Leaf")
            st.image(
                image,
                caption="Image submitted for analysis",
                width="stretch",
            )

        with right:
            st.subheader("🔍 AI Prediction")

            readable_name = predicted_class.replace(
                "___", " — "
            ).replace("_", " ")

            st.metric("Predicted Class", readable_name)
            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

            if confidence < 60:
                st.warning(
                    "Low confidence: treat this result cautiously. "
                    "Try a clear, close-up leaf image and seek "
                    "expert confirmation."
                )
            elif confidence < 80:
                st.info(
                    "Moderate confidence: compare the result with "
                    "the plant's actual symptoms."
                )
            else:
                st.success(
                    "The model strongly favours this class, "
                    "but the prediction still needs verification."
                )

        # -------------------------------------------------
        # 7. DISEASE DETAILS
        # -------------------------------------------------

        st.divider()
        st.header("🦠 Disease Information")

        if is_healthy:
            st.success("The model predicts a healthy leaf.")
        else:
            st.warning(
                "The model predicts a possible disease class. "
                "This is not a confirmed diagnosis."
            )

        st.subheader("What does this prediction mean?")
        st.write(disease_info["description"])

        st.subheader("Possible Symptoms")
        st.write(disease_info["symptoms"])

        # -------------------------------------------------
        # 8. PREVENTION SUGGESTIONS
        # -------------------------------------------------

        st.header("🛡️ Prevention and Care Suggestions")

        if is_healthy:
            st.write(
                "The model predicts a healthy class. "
                "Use these steps to help maintain plant health:"
            )
        else:
            st.write(
                "These general measures may help reduce disease "
                "spread or future risk. The appropriate response "
                "depends on the actual crop and confirmed disease."
            )

        for suggestion in disease_info["prevention"]:
            st.markdown(f"- {suggestion}")

        if not is_healthy:
            st.info(
                "Do not apply pesticides or fungicides based only "
                "on this AI prediction. Confirm the disease and "
                "follow local agricultural recommendations and "
                "the product label."
            )

        # -------------------------------------------------
        # 9. TOP THREE PREDICTIONS
        # -------------------------------------------------

        st.divider()
        st.header("📊 Top 3 Model Predictions")

        top_indices = np.argsort(predictions)[::-1][:3]

        for index in top_indices:
            class_name = class_names[int(index)]
            score = float(predictions[index]) * 100
            readable_class = class_name.replace(
                "___", " — "
            ).replace("_", " ")

            st.write(f"**{readable_class}** — {score:.2f}%")
            st.progress(min(max(score / 100, 0.0), 1.0))

        # -------------------------------------------------
        # 10. DOWNLOAD PREDICTION REPORT
        # -------------------------------------------------

        report = f"""
CROP DISEASE DETECTION REPORT
=============================

Predicted class: {readable_name}
Model confidence: {confidence:.2f}%

Description:
{disease_info['description']}

Possible symptoms:
{disease_info['symptoms']}

Prevention suggestions:
""" + "\n".join(
            f"- {item}" for item in disease_info["prevention"]
        ) + """

IMPORTANT:
This is an AI-generated prediction, not a confirmed diagnosis.
Consult a qualified agricultural expert before treatment.
"""

        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name="crop_disease_report.txt",
            mime="text/plain",
        )

    except Exception as error:
        st.error(f"Could not process the image: {error}")

else:
    st.warning("Upload a JPG or PNG leaf image to begin.")

# =========================================================
# 11. FOOTER
# =========================================================

st.divider()
st.caption(
    "Academic project prototype | MobileNetV2 | PlantVillage "
    "| Predictions should be verified before treatment."
)
