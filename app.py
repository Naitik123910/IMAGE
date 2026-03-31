from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf


# ------------------------------
# App-level constants
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = ["model.h5"]
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 520
MNIST_SIZE = (28, 28)


# ------------------------------
# Utility helpers
# ------------------------------
def resolve_model_path() -> Optional[Path]:
    """Return model.h5 path from current directory, if present."""
    for candidate in MODEL_CANDIDATES:
        p = BASE_DIR / candidate
        if p.exists():
            return p

    return None


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path) -> tf.keras.Model:
    """Load and cache Keras model to avoid reloading on every rerun."""
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_to_mnist(pil_img: Image.Image) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Preprocess input exactly in MNIST style:
    - grayscale
    - resize to 28x28
    - normalize to 0..1

    Returns:
      model_input_img: (28, 28) float32 image normalized to [0,1]
      display_img: uint8 image used for UI display
      is_blank: whether the processed image appears blank
    """
    gray = pil_img.convert("L")
    resized = gray.resize(MNIST_SIZE, Image.Resampling.LANCZOS)

    arr = np.array(resized).astype(np.float32) / 255.0

    # If background is bright (common for uploaded paper digits), invert to match MNIST.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Lightweight blank check for both canvas and uploads.
    is_blank = bool(np.max(arr) < 0.08 or np.std(arr) < 0.01)

    display_img = (arr * 255).astype(np.uint8)
    return arr, display_img, is_blank


def make_model_input(img_28: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """Adapt preprocessed 28x28 image to expected model input shape."""
    input_shape = model.input_shape

    # Handle multi-input models by using the first input shape.
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    # Common MNIST model shapes:
    # (None, 28, 28), (None, 28, 28, 1), (None, 1, 28, 28), (None, 784)
    if len(input_shape) == 2:
        return img_28.reshape(1, -1).astype(np.float32)

    if len(input_shape) == 3:
        return img_28.reshape(1, 28, 28).astype(np.float32)

    if len(input_shape) == 4:
        # Channels-last
        if input_shape[-1] == 1:
            return img_28.reshape(1, 28, 28, 1).astype(np.float32)
        # Channels-first
        if input_shape[1] == 1:
            return img_28.reshape(1, 1, 28, 28).astype(np.float32)

    # Safe fallback if model shape is unusual.
    return img_28.reshape(1, 28, 28, 1).astype(np.float32)


def predict_digit(model: tf.keras.Model, img_28: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Run prediction and return predicted digit, confidence and full probabilities."""
    model_input = make_model_input(img_28, model)
    raw_pred = model.predict(model_input, verbose=0)

    # Support outputs like (1, 10) and edge outputs that flatten to 10.
    probs = np.array(raw_pred).squeeze().astype(np.float32)

    if probs.ndim == 0:
        raise ValueError("Model output is scalar; expected 10-class probabilities.")

    if probs.ndim > 1:
        probs = probs.flatten()

    if probs.shape[0] != 10:
        raise ValueError(
            f"Model output has {probs.shape[0]} classes; expected 10 classes for digits 0-9."
        )

    # Convert logits to probabilities if needed.
    if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 1.0, atol=1e-2):
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / np.sum(exp_probs)

    pred_digit = int(np.argmax(probs))
    confidence = float(probs[pred_digit])
    return pred_digit, confidence, probs


def pil_from_canvas_rgba(canvas_rgba: np.ndarray) -> Image.Image:
    """Convert canvas RGBA array from streamlit-drawable-canvas to PIL image."""
    rgba = canvas_rgba.astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    # Flatten alpha safely for consistent grayscale conversion.
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    composite = Image.alpha_composite(bg, img)
    return composite.convert("RGB")


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✍",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f7fbff 0%, #eef6f7 100%);
        }
        .title-wrap {
            border-radius: 18px;
            padding: 18px 22px;
            background: linear-gradient(120deg, #0f766e 0%, #1d4ed8 100%);
            color: #ffffff;
            margin-bottom: 16px;
            box-shadow: 0 10px 22px rgba(15, 118, 110, 0.2);
        }
        .card {
            background: #ffffff;
            border: 1px solid #d8e5ea;
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 4px 14px rgba(16, 24, 40, 0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-wrap">
        <h2 style="margin:0;">Handwritten Digit Recognition</h2>
        <p style="margin:6px 0 0 0; opacity:0.95;">Draw or upload a digit image and get real-time MNIST-style prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload an image")
    uploaded_file = st.file_uploader(
        "Upload a digit image (png, jpg, jpeg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

model_path = resolve_model_path()
if model_path is None:
    detected_h5 = sorted([p.name for p in BASE_DIR.glob("*.h5")])
    detected_msg = f" Found: {', '.join(detected_h5)}" if detected_h5 else ""
    st.error(
        "model.h5 not found. Place your MNIST digit model as model.h5 in the app folder and rerun."
        + detected_msg
    )
    st.stop()

try:
    model = load_model(model_path)
except Exception as ex:
    st.error(f"Failed to load model: {ex}")
    st.stop()

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

left_spacer, center_canvas, right_spacer = st.columns([1, 3, 1])

with center_canvas:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Draw a digit")

    clear_clicked = st.button("Clear Canvas", use_container_width=True)
    if clear_clicked:
        st.session_state.canvas_key += 1
        st.rerun()

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#ffffff",
        background_color="#000000",
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="freedraw",
        key=f"digit_canvas_{st.session_state.canvas_key}",
        update_streamlit=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
predict_clicked = st.button("Predict Digit", type="primary", use_container_width=True)

if predict_clicked:
    source_img = None

    # Upload has higher priority if provided.
    if uploaded_file is not None:
        try:
            source_img = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Invalid image file. Please upload a valid PNG/JPG/JPEG image.")
            st.stop()
    elif canvas_result.image_data is not None:
        source_img = pil_from_canvas_rgba(canvas_result.image_data)

    if source_img is None:
        st.warning("No input found. Draw a digit or upload an image first.")
        st.stop()

    img_28, processed_display, is_blank = preprocess_to_mnist(source_img)

    if is_blank:
        st.warning("The input appears blank. Please draw or upload a clearer digit.")
        st.stop()

    try:
        pred_digit, confidence, probs = predict_digit(model, img_28)
    except Exception as ex:
        st.error(f"Prediction failed: {ex}")
        st.stop()

    left_res_spacer, center_results, right_res_spacer = st.columns([1, 3, 1])

    with center_results:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction")
        st.metric("Predicted Digit", str(pred_digit))
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Probability distribution (0-9)")
        prob_df = pd.DataFrame(
            {
                "Digit": list(range(10)),
                "Probability": probs,
            }
        )
        st.bar_chart(prob_df.set_index("Digit"), height=330)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Processed 28x28 image")
        st.image(processed_display, clamp=True, caption="Input used by the model", width=220)
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("Ready for local run and Render deployment.")
