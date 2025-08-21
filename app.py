import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Print TensorFlow version
st.write("Using TensorFlow:", tf.__version__)

# Load Models
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("best_model.h5")  # saved CNN
    hybrid_model = tf.keras.models.load_model("hybrid_model_n.h5")  # saved CNN+MLP
    return cnn_model, hybrid_model

cnn_model, hybrid_model = load_models()

# Preprocessing function
def preprocess_image(image, target_size=(128, 128)):  # adjust size as per your training
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    return img_array

# Streamlit UI
st.title("🐶🐱 CNN vs Hybrid Model Classifier")

uploaded_file = st.file_uploader("Upload an image (Cat/Dog)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)

    # CNN Prediction
    cnn_pred = cnn_model.predict(img_array, verbose=0)
    cnn_class = np.argmax(cnn_pred, axis=1)[0]

    # Hybrid Prediction
    hybrid_pred = hybrid_model.predict(img_array, verbose=0)
    hybrid_class = np.argmax(hybrid_pred, axis=1)[0]

    class_names = ["Cat", "Dog"]  # Change according to your dataset

    st.subheader("🔮 Predictions")
    st.write(f"**CNN Model:** {class_names[cnn_class]} (confidence: {np.max(cnn_pred):.2f})")
    st.write(f"**Hybrid Model:** {class_names[hybrid_class]} (confidence: {np.max(hybrid_pred):.2f})")
