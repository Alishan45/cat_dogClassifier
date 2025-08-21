import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import requests

# =========================
# MUST be the first Streamlit command
# =========================
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout="wide")

# =========================
# Utility for loading animations
# =========================
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# =========================
# Load Lottie Animations (replace with new URLs if needed)
# =========================
cat_animation = load_lottie_url("https://lottie.host/f34f6b84-c2e3-46df-944d-49e3b63a9b85/QyVb7KydOq.json")
dog_animation = load_lottie_url("https://lottie.host/67d91758-6f8e-4e69-9965-54f70e3aa10f/dsPkK8Rdp8.json")

# =========================
# Print TensorFlow version in sidebar
# =========================
st.sidebar.success(f"✅ Using TensorFlow {tf.__version__}")

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("best_model.h5")   # saved CNN
    hybrid_model = tf.keras.models.load_model("hybrid_model_n.h5")  # saved CNN+MLP
    return cnn_model, hybrid_model

cnn_model, hybrid_model = load_models()

# =========================
# Preprocessing function
# =========================
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    return img_array

# =========================
# Streamlit UI
# =========================
st.title("🐶🐱 CNN vs Hybrid Model Classifier")
st.markdown("### Upload an image and let AI decide: **Cat or Dog?**")

uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Predictions
    cnn_pred = cnn_model.predict(img_array, verbose=0)
    cnn_class = int(cnn_pred[0][0] > 0.5)  # sigmoid → binary

    hybrid_pred = hybrid_model.predict(img_array, verbose=0)
    hybrid_class = int(hybrid_pred[0][0] > 0.5)

    class_names = ["Cat", "Dog"]

    # =========================
    # Results Section
    # =========================
    st.subheader("🔮 AI Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 CNN Model")
        st.metric("Prediction", class_names[cnn_class], f"{cnn_pred[0][0]*100:.1f}%")
        st.progress(float(cnn_pred[0][0]))

    with col2:
        st.markdown("### 🤖 Hybrid CNN+MLP Model")
        st.metric("Prediction", class_names[hybrid_class], f"{hybrid_pred[0][0]*100:.1f}%")
        st.progress(float(hybrid_pred[0][0]))

    # =========================
    # Fun Animations
    # =========================
    st.markdown("---")
    st.subheader("✨ Fun Visualization")

    if cnn_class == 0 or hybrid_class == 0:  # Cat predicted
        if cat_animation:
            st_lottie(cat_animation, height=250, key="cat")
        else:
            st.warning("🐱 Cat animation could not be loaded.")
    else:
        if dog_animation:
            st_lottie(dog_animation, height=250, key="dog")
        else:
            st.warning("🐶 Dog animation could not be loaded.")

# =========================
# Sidebar Info
# =========================
st.sidebar.header("📌 About This App")
st.sidebar.info(
    """
    This app compares two deep learning models:  
    - 🧠 **CNN Model**  
    - 🤖 **Hybrid CNN + MLP Model**  

    Upload a **Cat/Dog image** and see which model performs better!  
    """
)
