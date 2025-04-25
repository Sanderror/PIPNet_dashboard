import streamlit as st
import os
import shutil
import subprocess
from pathlib import Path
import re
import random

USER_IMAGES_DIR = Path("data/user_images/uploaded_images")
RESULTS_DIR = Path("runs/run_pipnet/visualization_results")
CLASS_DIR = Path("data/CUB_200_2011/dataset/train")

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

    predictions = parse_predictions()
    i = 1
    for img in USER_IMAGES_DIR.iterdir():
        print(img)
        img_name = img.name
        st.header(f"🖼️ Image {i}: {img_name}")
        img_scraped = re.sub(r'\.(jpg|jpeg|png)$', '', img_name, flags=re.IGNORECASE)
        preds = predictions[img_scraped]
        st.image(f"{USER_IMAGES_DIR}/{img_name}", caption=f"Uploaded image", width=256)
        st.subheader(f"🔎 Top 3 Predictions")
        for cls, score in preds:
            st.write(f"🔹 {cls} — {score:.3f}")
            class_dir = os.path.join(CLASS_DIR, cls)
            class_samples = [f for f in os.listdir(class_dir)]
            random_sample = random.choice(class_samples)
            st.image(f"{class_dir}/{random_sample}", caption=f"Sample image for class {cls}", width=128)
        i += 1
