import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

MODEL_PATH = Path("model.h5")
CLASS_FILE = Path("class_names.json")
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Zatürre Tespit Sistemi", page_icon="🫁", layout="centered")
st.title("Derin Öğrenme Tabanlı Akciğer Röntgeninden Zatürre Tespiti")
st.write(
    "Kullanıcı akciğer röntgen görüntüsünü yükler. Sistem görüntüyü analiz ederek "
    "**Normal** veya **Zatürre** tahmini üretir."
)
st.warning("Bu uygulama eğitim amaçlı karar destek prototipidir. Klinik tanı amacıyla tek başına kullanılamaz.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if CLASS_FILE.exists():
        with open(CLASS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["NORMAL", "PNEUMONIA"]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Model 224x224 RGB görüntü bekler. Piksel ölçekleme modelin içinde yapılır."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.asarray(image).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_label(model, image: Image.Image, class_names):
    x = preprocess_image(image)
    pred = model.predict(x, verbose=0)

    # Binary sigmoid model: output shape (1, 1)
    if pred.shape[-1] == 1:
        pneumonia_prob = float(pred[0][0])
        if pneumonia_prob >= 0.5:
            return "ZATÜRRE", pneumonia_prob
        return "NORMAL", 1.0 - pneumonia_prob

    # Softmax model: output shape (1, 2)
    idx = int(np.argmax(pred[0]))
    label = class_names[idx]
    confidence = float(pred[0][idx])
    if label.upper() == "PNEUMONIA":
        label = "ZATÜRRE"
    return label, confidence


model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader("Röntgen görüntüsü yükle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Yüklenen Görüntü")
    st.image(image, use_container_width=True)

    if model is None:
        st.error("model.h5 bulunamadı. Önce train_model.py ile modeli eğitip model.h5 dosyasını oluşturmalısın.")
        st.code("python train_model.py --data_dir data/chest_xray --epochs 10")
    else:
        label, confidence = predict_label(model, image, class_names)
        st.subheader("Tahmin Sonucu")
        if label == "ZATÜRRE":
            st.error(f"Sonuç: {label}")
        else:
            st.success(f"Sonuç: {label}")
        st.metric("Tahmin güveni", f"%{confidence * 100:.2f}")

st.divider()
st.subheader("Model Performans Çıktıları")
for graph_path in [Path("grafikler/accuracy_loss.png"), Path("grafikler/confusion_matrix.png")]:
    if graph_path.exists():
        st.image(str(graph_path), caption=graph_path.name, use_container_width=True)
    else:
        st.info(f"{graph_path} henüz oluşturulmadı. Model eğitimi tamamlanınca üretilecek.")
