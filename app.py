"""Streamlit demo: upload a photo, get the model's guess at which UCSD
building it is.

    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent / "Model"))
from model import net, eval_transform, TRAIN_DIR, WEIGHTS  # noqa: E402

st.set_page_config(page_title="UCSD Building Classifier", page_icon="\U0001F3DB️")
st.title("UCSD Building Classifier")
st.write("Upload a photo of a UCSD building and the model will guess which one it is.")


@st.cache_resource
def load_classifier():
    classes = datasets.ImageFolder(TRAIN_DIR).classes
    net.initialize()
    net.load_params(f_params=WEIGHTS)
    return classes


classes = load_classifier()

uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your photo", use_container_width=True)

    tensor = eval_transform(image).unsqueeze(0)
    probs = net.predict_proba(tensor)[0]
    pred = probs.argmax()

    st.subheader(f"Prediction: {classes[pred].replace('_', ' ')}")
    st.write(f"Confidence: {probs[pred] * 100:.1f}%")

    st.write("All probabilities:")
    for name, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
        st.write(f"{name.replace('_', ' ')}: {p * 100:.1f}%")
