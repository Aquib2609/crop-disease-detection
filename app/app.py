import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import cv2

# ==================================================
# Resolve BASE directory safely
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# Load trained model (ABSOLUTE PATH)
# ==================================================
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "mobilenetv2_crop_disease.h5"
)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==================================================
# Load class names safely
# ==================================================
DATASET_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "..",
    "dataset",
    "PlantVillage",
    "train"
)

if not os.path.exists(DATASET_TRAIN_PATH):
    st.error("❌ Dataset folder not found for loading class names.")
    st.stop()

CLASS_NAMES = [
    name.replace("___", " - ").replace("_", " ")
    for name in sorted(os.listdir(DATASET_TRAIN_PATH))
]

# ==================================================
# Label normalization (CRITICAL FIX)
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
# Disease descriptions & remedies (NORMALIZED KEYS)
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
st.set_page_config(
    page_title="Crop Disease Detection",
    layout="centered"
)

st.title("🌱 Crop Disease Detection")
st.markdown(
    "Upload a **leaf image** to predict the **crop disease**, view **confidence scores**, "
    "see **possible treatments**, and understand **model attention using Grad-CAM**."
)

uploaded_file = st.file_uploader(
    "📤 Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# Image preprocessing
# ==================================================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==================================================
# Grad-CAM utilities
# ==================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    img = np.array(image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    return np.uint8(superimposed_img)

# ==================================================
# Prediction logic
# ==================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    processed_image = preprocess_image(image)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing leaf... 🌿"):
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

        # ==================================================
        # Grad-CAM Visualization
        # ==================================================
        st.subheader("🔥 Model Attention (Grad-CAM)")

        last_conv_layer_name = "Conv_1"  # MobileNetV2 last conv layer
        heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)
        gradcam_image = overlay_gradcam(image, heatmap)

        st.image(
            gradcam_image,
            caption="Grad-CAM Heatmap (highlighted regions influenced the prediction)",
            use_column_width=True
        )
