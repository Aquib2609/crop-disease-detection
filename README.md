🌱 Crop Disease Detection using Deep Learning

An end-to-end, explainable AI application for detecting plant leaf diseases from images using Deep Learning and Computer Vision.
The system allows users to upload a leaf image and receive top-3 disease predictions with confidence scores, disease descriptions & remedies, and Grad-CAM heatmaps to visually explain the model’s decision.


🚀 Live Demo

🔗 Streamlit Cloud :https://crop-disease-detection-fs4zyunykbg4fmuuef6l4h.streamlit.app/


📌 Key Features

✅ Image-based crop disease detection

✅ Top-3 predictions with confidence scores

✅ Explainable AI using Grad-CAM heatmaps

✅ Disease descriptions and remedies

✅ Streamlit-based interactive web app

✅ Lightweight MobileNetV2 architecture

✅ Deployed on Hugging Face Spaces


🧠 Model Overview

Architecture: MobileNetV2 (Transfer Learning)

Framework: TensorFlow / Keras

Input Size: 224 × 224 RGB images

Dataset: PlantVillage (used during training)

Validation Accuracy: ~94%

The model was fine-tuned to classify 38 different crop–disease categories, achieving strong generalization and robustness.


📊 Explainability with Grad-CAM

To ensure transparency and trust, the application uses Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight regions of the leaf image that influenced the model’s prediction.

This helps verify that the model focuses on disease-affected areas rather than background noise.


🧪 Example Workflow

Upload a plant leaf image

Model analyzes the image

Top-3 disease predictions are displayed with confidence

Disease description & suggested remedies are shown

Grad-CAM heatmap visualizes model attention
