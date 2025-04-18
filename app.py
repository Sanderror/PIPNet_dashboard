import streamlit as st
import os
import shutil
import subprocess
from pathlib import Path

USER_IMAGES_DIR = Path("data/user_images/uploaded_images")
RESULTS_DIR = Path("runs/run_pipnet/visualization_results")

# Utility functions
def clear_folder(folder):
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def save_uploaded_files(uploaded_files):
    clear_folder(USER_IMAGES_DIR)
    USER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(USER_IMAGES_DIR / uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

def run_prediction_script():
    subprocess.run(["python", "main.py", "--state_dict_dir_net", "./checkpoints/net_trained"], check=True)

def parse_predictions():
    results = {}
    for image_folder in RESULTS_DIR.iterdir():
        if image_folder.is_dir():
            preds = []
            for class_folder in image_folder.iterdir():
                if class_folder.is_dir():
                    name_score = class_folder.name
                    if "_" in name_score:
                        cls, score = name_score.rsplit("_", 1)
                        try:
                            preds.append((cls, float(score)))
                        except ValueError:
                            continue
            preds.sort(key=lambda x: -x[1])
            results[image_folder.name] = preds[:3]
    return results

# Streamlit UI
st.title("🧠 PIPNet Image Classifier Dashboard")

uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    save_uploaded_files(uploaded_files)
    st.success(f"{len(uploaded_files)} image(s) saved.")

if st.button("Predict"):
    with st.spinner("Running model..."):
        run_prediction_script()
    st.success("Prediction complete!")

    st.subheader("🔎 Top 3 Predictions")
    predictions = parse_predictions()
    for image_id, preds in predictions.items():
        st.markdown(f"**{image_id}**")
        for cls, score in preds:
            st.write(f"🔹 {cls} — {score:.3f}")
