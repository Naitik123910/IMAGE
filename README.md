# IMAGE -> Enviroment
# Handwritten Digit Recognition Web App

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="#"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white"></a>
  <a href="#"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

A production-ready **Streamlit** web app for recognizing handwritten digits (**0-9**) using a trained **TensorFlow/Keras** model (`model.h5`).

The app supports:
- Drawing digits on a canvas
- Uploading image files
- MNIST-style preprocessing (grayscale, 28x28 resize, normalization)
- Prediction with confidence score
- Full probability distribution chart for classes 0-9

---

## Preview

- **Input methods**: Canvas + Image upload
- **Outputs**: Predicted digit, confidence, processed image, probability bars
- **UX**: Sidebar instructions, modern UI cards, clear canvas button, edge-case handling

> Tip: Add screenshots or GIFs in a `docs/` folder and link them here for an even better GitHub presentation.

---

## Project Structure

```text
IMAGE/
├── app.py
├── model.h5                  # Your MNIST-trained model (required)
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

> Current workspace contains `plant_disease_model.h5`. For digit recognition, place your MNIST model as `model.h5`.

---

## Features

- Interactive drawing canvas for digit input (0-9)
- Image upload support (`.png`, `.jpg`, `.jpeg`)
- Exact MNIST-style preprocessing pipeline:
  - Convert to grayscale
  - Resize to `28x28`
  - Normalize to range `[0, 1]`
- Robust model loading with deployment-safe relative paths
- Prediction output:
  - Predicted digit
  - Confidence score
  - Probability distribution bar chart (0-9)
- Processed image preview (actual model input)
- Edge-case handling:
  - Blank canvas
  - Invalid image uploads
  - Unexpected model output shape

---

## Tech Stack

- **Python**
- **Streamlit**
- **TensorFlow / Keras**
- **Pillow**
- **NumPy / Pandas**
- **streamlit-drawable-canvas**

---

## Quick Start (Local)

### 1) Clone and enter project folder

```bash
git clone <your-repo-url>
cd IMAGE
```

### 2) (Recommended) Create virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Add your model

Place your trained MNIST model file in the project root and name it:

```text
model.h5
```

### 5) Run app

```bash
streamlit run app.py
```

If `streamlit` command is not found:

```bash
python -m streamlit run app.py
```

---

## Deploy on Render

This repo is configured for Render using:
- `requirements.txt`
- `Procfile`
- `runtime.txt`

### Steps

1. Push this project to GitHub.
2. In Render, click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Set configuration:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Ensure `model.h5` is included in the repository root.
6. Click **Create Web Service**.

Render will assign a public URL for your app.

---

## Configuration Notes

- App resolves model path relative to `app.py`.
- Preferred model name: `model.h5`.
- If `model.h5` is missing, app falls back to the first `.h5` file found.

For correct digit predictions, always keep your MNIST model as `model.h5`.

---

## Troubleshooting

### 1) Wrong predictions

- Verify correct model (`model.h5`) is present.
- Draw thicker strokes and keep one clear digit centered.
- Use high-contrast uploaded images.

### 2) App says input is blank

- Draw with more visible strokes on canvas.
- Upload a clearer image with strong foreground/background contrast.

### 3) Import errors in editor

This usually means dependencies are not installed in the active environment.

```bash
pip install -r requirements.txt
```

### 4) Render build issues

- Confirm Python runtime from `runtime.txt`
- Confirm all required packages in `requirements.txt`
- Confirm `model.h5` exists in repo root

---

## Future Enhancements

- Add confidence threshold warnings
- Add top-3 prediction display
- Add drag-and-drop image support
- Add model versioning and experiment metadata
- Add unit tests for preprocessing and inference

---

## License

This project is available under the **MIT License**.
You can add a `LICENSE` file to the repository if needed.
