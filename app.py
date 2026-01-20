import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from huggingface_hub import hf_hub_download

# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="Crop Disease Detection",
    layout="centered"
)

st.title("🌱 Crop Disease Detection")
st.markdown(
    "Upload a **leaf image** to predict the **crop disease**, view **confidence scores**, "
    "and see **possible treatments**."
)

# ==================================================
# Load class names from classes.txt (CRITICAL FIX)
# ==================================================
CLASSES_PATH = "classes.txt"

if not os.path.exists(CLASSES_PATH):
    st.error("❌ classes.txt file not found in repository.")
    st.stop()

with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# ==================================================
# Lazy load model from Hugging Face Model Hub
# ==================================================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Aquib2609/Crop-disease-detection",
        filename="mobilenetv2_crop_disease.h5"
    )
    return tf.keras.models.load_model(model_path,compile=False)

model = None  # Do NOT load at startup

# ==================================================
# Image preprocessing
# ==================================================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==================================================
# Label normalization (for disease info mapping)
# ==================================================
def normalize_label(label: str) -> str:
    return (
        label.lower()
        .replace(",", "")
        .replace("-", "")
        .replace("  ", " ")
        .strip()
    )

# ==================================================
# Disease descriptions & remedies (OPTIONAL INFO)
# ==================================================
DISEASE_INFO = {
    normalize_label("Potato - Early blight"): {
        "description": "Early blight is a fungal disease caused by Alternaria solani, producing dark concentric lesions on leaves.",
        "remedy": "Use certified seeds, apply fungicides, remove infected leaves, and practice crop rotation."
    },
    normalize_label("Pepper, bell - Bacterial spot"): {
        "description": "Bacterial spot causes water-soaked lesions that become brown and scabby in warm, humid conditions.",
        "remedy": "Apply copper-based sprays, avoid overhead irrigation, and remove infected plants."
    },
    normalize_label("Tomato - Late blight"): {
        "description": "Late blight is a rapidly spreading disease that causes dark lesions and plant decay.",
        "remedy": "Use resistant varieties and apply fungicides preventively."
    }
}

# ==================================================
# Streamlit UI
# ==================================================
uploaded_file = st.file_uploader(
    "📤 Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# Prediction logic
# ==================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    processed_image = preprocess_image(image)

    if st.button("🔍 Predict Disease"):

        with st.spinner("Downloading model (first time) and analyzing leaf... 🌿"):

            # Lazy load model only when needed
            if model is None:
                model = load_model()

            preds = model.predict(processed_image)[0]

        # Top-3 predictions
        top_indices = np.argsort(preds)[-3:][::-1]

        st.subheader("🔍 Top Predictions")

        for i, idx in enumerate(top_indices):
            disease = CLASS_NAMES[idx]
            confidence = preds[idx] * 100

            if i == 0:
                st.success(f"🥇 **{disease}** — {confidence:.2f}%")
            else:
                st.write(f"**{disease}** — {confidence:.2f}%")

            st.progress(float(preds[idx]))

            # Show disease info only for top-1
            if i == 0:
                info = DISEASE_INFO.get(normalize_label(disease))
                if info:
                    st.markdown("### 🌾 Disease Information")
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Remedy:** {info['remedy']}")
                else:
                    st.info("ℹ️ Disease information not available yet.")

        
