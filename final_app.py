# --- IMPORTS ---
import streamlit as st
import subprocess
from pathlib import Path
import base64
import re
import streamlit.components.v1 as components
from PIL import Image
import torch
import shutil

# --- CONFIGURATION ---
st.markdown("""
<style>
div[role="radiogroup"] > label {
    background-color: #f0f2f6;
    border: 1px solid #d9d9d9;
    border-radius: 10px;
    padding: 8px 16px;
    margin-right: 10px;
    cursor: pointer;
}
div[role="radiogroup"] > label[data-selected="true"] {
    background-color: #3399FF;
    color: white;
    border: 1px solid #3399FF;
}
</style>
""", unsafe_allow_html=True)

# --- FOLDERS ---
USER_IMAGES_DIR = Path("data/user_images/uploaded_images")
RESULTS_DIR = Path("runs/run_pipnet/visualization_results")
CLASS_IMAGES_DIR = Path("data/CUB_200_2011/dataset/train")
PROTOTYPE_DIR = Path("visualised_prototypes")
DASHBOARD_IMAGES_DIR = Path("dashboard_images")
NET_WEIGHTS_DIR = Path('runs/run_pipnet/net_weights.pt')
ORIG_NET_WEIGHTS_DIR = Path('runs/run_pipnet/original_net_weights.pt')

# --- SESSION STATE ---
st.session_state.setdefault("current_tab", "About")
st.session_state.setdefault("predictions", None)
st.session_state.setdefault("uploaded", False)

# --- UTILITY FUNCTIONS ---
def clear_folder(folder):
    '''Clears a folder, which is used to remove previous user uploaded images'''
    if folder.exists():
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

def save_uploaded_files(uploaded_files):
    '''This is used to save the uploaded images to the correct directory'''
    clear_folder(USER_IMAGES_DIR)
    USER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(USER_IMAGES_DIR / uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

def run_prediction_script():
    '''Function that runs the main.py file in order to obtain the predictions on the uploaded images'''
    subprocess.run(["python", "main.py", "--state_dict_dir_net", "./checkpoints/net_trained"], check=True)

def get_all_class_images(class_name):
    '''Function that will return a list with the uploaded images from the user'''
    for folder in CLASS_IMAGES_DIR.iterdir():
        if folder.is_dir() and class_name in folder.name:
            return list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    return []

def extract_patch_info(filename):
    '''Function to extract patch information from prototype explanation images names (mainly for the simalarity score)'''
    try:
        return {
            "mul": float(re.search(r"mul([0-9.]+)", filename).group(1)) if re.search(r"mul([0-9.]+)", filename) else None,
            "sim": float(re.search(r"sim([0-9.]+)", filename).group(1)) if re.search(r"sim([0-9.]+)", filename) else None,
            "w": float(re.search(r"w([0-9.]+)", filename).group(1)) if re.search(r"w([0-9.]+)", filename) else None,
        }
    except Exception:
        return {"mul": None, "sim": None, "w": None}

def image_to_base64(img_path):
    '''Transform images'''
    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

def show_image_carousel(image_paths, height=400, caption=None):
    '''Creates a streamlit image carousel for input images (used for reference features and predicted class)'''
    if not image_paths:
        st.warning("No images to display.")
        return

    slides_html = "".join(
        f"<div class='slide'><img src='{image_to_base64(img)}' style='width:100%'></div>"
        for img in image_paths
    )

    caption_html = f"<div id='carousel-caption' style='text-align: center; font-size: 12px; color: gray; margin-top: 8px;'>{caption}</div>" if caption else ""

    carousel_html = f"""
    <style>
    .carousel-container {{
        position: relative;
        max-width: 100%;
        margin: auto;
        overflow: hidden;
    }}
    .slide {{ display: none; }}
    .slide img {{
        border-radius: 10px;
        max-height: {height}px;
        object-fit: contain;
    }}
    .active {{ display: block; }}
    .prev, .next {{
        cursor: pointer;
        position: absolute;
        top: 50%;
        padding: 16px;
        color: white;
        background-color: rgba(0,0,0,0.5);
        font-size: 18px;
    }}
    .next {{ right: 0; }}
    </style>

    <div class="carousel-container">
        {slides_html}
        <a class="prev" onclick="plusSlides(-1)">❮</a>
        <a class="next" onclick="plusSlides(1)">❯</a>
    </div>
    {caption_html}
    <div id="slideIndexLabel" style="text-align: center; font-size: 12px; color: gray; margin-top: 4px;"></div>
    <script>
    let slideIndex = 0;
    let slides = document.getElementsByClassName("slide");
    function showSlides(n) {{
        for (let i = 0; i < slides.length; i++) slides[i].style.display = "none";
        slides[n].style.display = "block";
        document.getElementById("slideIndexLabel").innerText = `${{slideIndex+1}} / ${{slides.length}}`;
    }}
    function plusSlides(n) {{
        slideIndex = (slideIndex + n + slides.length) % slides.length;
        showSlides(slideIndex);
    }}
    showSlides(slideIndex);
    </script>
    """

    components.html(carousel_html, height=height + 70)

def get_sorted_rect_images(image_folder, class_name):
    '''Function to obtain the images containing the image region in the uploaded image'''
    rect_images = []
    for folder in RESULTS_DIR.glob(f"{image_folder}/{class_name}*"):
        if folder.is_dir():
            for file in folder.glob("*_rect*"):
                try:
                    score = float(file.name.split("_")[0][3:])
                    rect_images.append((score, file))
                except Exception:
                    continue
    rect_images.sort(key=lambda x: -x[0])
    return [f for _, f in rect_images]

def get_prototype_folder(patch_filename):
    '''Function to obtain the prototype/reference feature id'''
    parts = patch_filename.split("_")
    for part in parts:
        if part.startswith("p") and part[1:].isdigit():
            return f"prototype_{part[1:]}"
    return None

def get_prototype_images(prototype_id):
    '''Function to obtain all images in a prototype folder (for the image caroussel)'''
    folder = PROTOTYPE_DIR / prototype_id
    if folder.exists() and folder.is_dir():
        return sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    return []

def show_centered_image(path, width=400, padding=20):
    '''Function to show the image nicely'''
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="text-align: center; padding-top: {padding}px; padding-bottom: {padding}px;">
        <img src="data:image/png;base64,{data}" width="{width}">
    </div>
    """, unsafe_allow_html=True)

def verbal_label(percentage):
    '''Function to convert the network's weights to importance levels for clarity to the user'''
    if percentage <= 10:
        return "🟦 Very Low"
    elif percentage <= 25:
        return "🟩 Low"
    elif percentage <= 50:
        return "🟨 Moderate"
    elif percentage <= 75:
        return "🟧 High"
    else:
        return "🟥 Very High"

def calculate_net_score(cls, cls_folder):
    '''Function to calculate the prediction score for an uploaded image'''
    total_cls_score = 0
    # Obtain the index
    cls_idx = int(cls.split(".")[0]) - 1
    net_weights = torch.load(NET_WEIGHTS_DIR).detach().clone()
    # Get all the weights of a class for every prototype
    cls_weights = net_weights[cls_idx]
    for file in cls_folder.glob("*_rect*"):
        try:
            # Use simarity from image name + weights in network to calculate all prediction strengths
            orig_mul, prot, sim, orig_weight, _ = file.name.split("_")
            prot_idx = int(prot.strip('p'))
            sim_score = float(sim.strip('sim'))
            net_weight = cls_weights[prot_idx].item()
            mul_score_updated = sim_score * net_weight
            total_cls_score += mul_score_updated
        except Exception:
            continue
    return total_cls_score

def parse_predictions():
    '''Function to obtain for each uploaded image the predictions (class, score)'''
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
                            # Calculate the score directly instead of obtaining it form image name
                            updated_score = calculate_net_score(cls, class_folder)
                            preds.append((cls, updated_score))
                        except ValueError:
                            continue
            preds.sort(key=lambda x: -x[1])
            total_bird_strength = sum([scr for cls, scr in preds])
            # Only return the top 3 bird predictions
            results[image_folder.name] = (preds[:3], total_bird_strength)
    return results

def render_predictions_tab():
    '''Function to create the predictions page'''
    # parse predictions every time model is reran, so the prediction confidences are updated
    predictions_dct = parse_predictions()

    # Select the image you would like the predictions to see for
    image_options = list(predictions_dct.keys())
    st.header("🔎 Choose an image to explore its predictions")
    selected_image_id = st.selectbox("Select one of your uploaded images:", image_options)

    # Obtain the predictions for the selected image
    preds, total_bird_strength = predictions_dct[selected_image_id]
    matches = list(USER_IMAGES_DIR.glob(f"{selected_image_id}.*"))
    image_path = matches[0] if matches else None

    # Show the uploaded image once more
    st.subheader("3️⃣ Top 3 predicted bird species for your image")
    if image_path and image_path.exists():
        st.image(str(image_path), caption="Your uploaded image", width=400)
    st.markdown("---")

    # Go through all top 3 predictions and create a separate section for each
    for prediction_idx, (cls, score) in enumerate(preds, start=1):
        display_cls = " ".join(cls.split(".")[-1].split("_"))
        cls_index = int(cls.split(".")[0]) - 1

        # Calculate prediction confidence
        pred_confidence = round(score / total_bird_strength * 100, 2)
        st.markdown(f"### {prediction_idx}. {display_cls} — Prediction Confidence: {pred_confidence:.2f}%")

        # Create a caroussel showcasing the predicted bird with multiple images of it
        class_images = get_all_class_images(cls)
        if class_images:
            show_image_carousel(class_images, caption=f"Images of {display_cls}")
            rects = get_sorted_rect_images(selected_image_id, cls)
            st.markdown("---")
        else:
            st.warning(f"No images found for {display_cls}")

        if rects:
            # Then we go through each prototype to create a separate section for each prototype
            for idx, img_path in enumerate(rects, start=1):
                # Obtain some features
                prototype_id = get_prototype_folder(img_path.name)
                prototype_num = prototype_id.replace("prototype_", "") if prototype_id else "?"
                prototype_imgs = get_prototype_images(prototype_id)
                info = extract_patch_info(img_path.name)
                fixed_height = 160

                # Create a section with 3 columns where the image patch is shown, the reference feature, and the statistics
                col_patch, col_proto, col_stats = st.columns(3)
                with col_patch:
                    st.image(str(img_path), caption=f"Image section #{idx}", width=fixed_height)
                with col_proto:
                    if prototype_imgs:
                        show_image_carousel(prototype_imgs, caption=f"Images of Reference Feature #{prototype_num}", height=160)
                    else:
                        st.warning("No prototype found")
                with col_stats:
                    if all(v is not None for v in info.values()):
                        # Obtain the updated network weights and calculate the scores of a prototype
                        net_weights = torch.load(NET_WEIGHTS_DIR).detach().clone()
                        updated_weight = float(net_weights[cls_index, int(prototype_num)])
                        updated_mul = updated_weight * info["sim"]
                        updated_percent = (updated_mul / score) * 100 if score else 0
                        prototype_weights = net_weights[:, int(prototype_num)]

                        # In order to get the importance level (we divide this class' weight by the total weight in the prototype)
                        total_prototype_weight = float(prototype_weights.sum())
                        percent_w = (
                            round(updated_weight / total_prototype_weight * 100)
                            if total_prototype_weight
                            else 0
                        )
                        verbal = verbal_label(percent_w)

                        # Add all this information in a markdown cell
                        st.markdown(
                            f"""
                            ##### **Reference Feature {prototype_num}**  
                            <div style="margin-top: 10px;">
                                <div><strong>• Similarity:</strong> {info['sim'] * 100:.1f}%</div>
                                <div><strong>• Importance:</strong> {verbal}</div>
                                <div><strong>• Contribution to prediction:</strong> {updated_percent:.1f}%</div>
                            </div>
                            <div style="background-color: #e0e0e0; width: 100%; height: 20px; border-radius: 5px; overflow: hidden; margin-top: 10px;">
                                <div style="background-color: #4CAF50; width: {updated_percent}%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 12px; color: white; font-weight: bold;">
                                    {updated_mul:.1f}
                                </div>
                            </div>
                            <div style="text-align: right; font-size: 12px; color: gray; margin-top: 4px;">
                                Total Prediction Strength: {score:.1f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Create the section for weight/importance adjusting
                # We again need to get the current network weights
                net_weights = torch.load(NET_WEIGHTS_DIR).detach().clone()
                prototype_weights = net_weights[:, int(prototype_num)]
                class_prototype_weight = float(net_weights[cls_index, int(prototype_num)])
                total_prototype_weight = float(prototype_weights.sum())

                # If the user changes the importance, it will be set to this value
                verbal_levels = {
                    "🟦 Very Low": 0.01, # Do not set this to exactly 0, otherwise cannot re-update weights
                    "🟩 Low": 0.25,
                    "🟨 Moderate": 0.5,
                    "🟧 High": 0.75,
                    "🟥 Very High": 0.99, # Do not set this to exactly 1, otherwise cannot re-update weights
                }

                # Obtain current percentage to initialize the slider with
                current_percentage = (
                    round(class_prototype_weight / total_prototype_weight * 100)
                    if total_prototype_weight
                    else 0
                )
                current_level = verbal_label(current_percentage)

                with st.expander(f"🛠️ Change the importance of this reference feature here!"):
                    # Create the slider
                    selected_verbal = st.select_slider(
                        f"Choose new importance for '{display_cls}' in Reference Feature {prototype_num}",
                        options=list(verbal_levels.keys()),
                        value=current_level,
                        key=f"verbal_slider_{selected_image_id}_{cls}_{prototype_num}",
                    )

                    # Calculate the updated network weight for the predicted class in the prototype
                    new_weight_value = verbal_levels[selected_verbal] * total_prototype_weight

                    # Calculate ratio to update other weights in prototype in order to update these correctly
                    new_weights_other_classes = total_prototype_weight - new_weight_value
                    old_weights_other_classes = total_prototype_weight - class_prototype_weight
                    if old_weights_other_classes > 0:
                        update_ratio_other_classes = new_weights_other_classes / old_weights_other_classes
                    else:
                        update_ratio_other_classes = 1

                    # Add a reset and apply button to handle the importance adjusting
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        reset_clicked = st.button(
                            "🔄 Reset to original",
                            key=f"reset_button_{selected_image_id}_{cls}_{prototype_num}",
                        )
                    with col2:
                        submit_clicked = st.button(
                            "🚀 Apply update",
                            key=f"submit_button_{selected_image_id}_{cls}_{prototype_num}",
                        )

                    # If the weight is reset, we take the original weight for the prototype and update the current network weights with it
                    if reset_clicked:
                        orig_weights = torch.load(ORIG_NET_WEIGHTS_DIR).detach().clone()
                        net_weights[:, int(prototype_num)] = orig_weights[:, int(prototype_num)]
                        torch.save(net_weights, NET_WEIGHTS_DIR)
                        st.success("✅ Importance has been reset to original value!")
                        st.rerun()

                    # if the apply button has been clicked, the new network weights will be calculated for the classes in the affected prototype
                    if submit_clicked:
                        # Update the other class weights of this prototype
                        for idx, w in enumerate(prototype_weights):
                            if idx != cls_index and w > 0:
                                new_weight = w * update_ratio_other_classes
                                net_weights[idx, int(prototype_num)] = new_weight
                        net_weights[cls_index, int(prototype_num)] = new_weight_value
                        torch.save(net_weights, NET_WEIGHTS_DIR)
                        st.success("✅ Updated based on new importance level!")
                        st.rerun()

                st.markdown(
                    "<hr style='margin: 25px 0; height: 0.1px; background-color: #ccc; border: none;'>",
                    unsafe_allow_html=True
                )                        

# --- UI ---
st.markdown("<h1 style='text-align: center;'>🐦 Bird Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'><em>See What the Model Sees – Powered by PIPNet</em></h5>", unsafe_allow_html=True)
st.markdown("---")

tabs = ["About", "Explanation", "Upload", "Predictions"]
selected_tab = st.radio(
    "",
    tabs,
    index=tabs.index(st.session_state.current_tab),
    horizontal=True
)
st.markdown("---")

# --- TAB CONTENT ---
if selected_tab == "About":
    st.header("📖 About")
    st.markdown("""
    Welcome to the PIPNet Dashboard!  
    - Upload bird images.
    - Predict and explore how the model makes decisions.
    - See image region-to-reference feature explanations.
    """)

elif selected_tab == "Upload":
    st.header("📤 Upload your images")
    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        save_uploaded_files(uploaded_files)
        st.session_state.uploaded = True
        st.success(f"{len(uploaded_files)} image(s) uploaded successfully!")
        for img in uploaded_files:
            encoded = base64.b64encode(img.getbuffer()).decode()
            st.markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                    <img src='data:image/png;base64,{encoded}' width='250'>
                    <div style='font-size: 14px; color: gray;'>{img.name}</div>
                </div>
            """, unsafe_allow_html=True)

    # Run the prediction script if button is clicked
    if st.button("Predict"):
        with st.spinner("Running model..."):
            run_prediction_script()
        st.session_state.predictions = "true"
        st.session_state.current_tab = "Predictions"
        st.rerun()

elif selected_tab == "Predictions":
    if not st.session_state.predictions:
        st.info("Please upload images and predict first.")
    else:
        render_predictions_tab()
        st.markdown("---")

elif selected_tab == "Explanation":
    st.header("❓ How to Interpret Predictions")

    st.markdown("#### 1️⃣ What does the prediction mean?")
    show_centered_image("dashboard_images/name_score_final.png", width=275, padding=30)
    st.markdown("Each prediction shows the **predicted class** and its **confidence**. A higher confidence means the model is more confident about the uploaded bird being that class. The confidence is calculated over 200 different bird classes, therefore even a score of **39.05%** is also already quite high.")

    st.markdown("#### 2️⃣ How can I tell if the prediction makes sense?")
    show_centered_image("dashboard_images/class_carousel_final.png", width=350, padding=30)
    st.markdown("""
    For each predicted class, we show a **carousel of typical images** from that bird species. This helps you compare:

    - Does your photo look similar to those examples?
    - Can you spot important visual features (color, beak shape, etc.)?

    If not, the model might be making a mistake — and that’s worth exploring.
    """)

    st.markdown("#### 3️⃣ What are Reference Features and Image Section Explanations?")
    show_centered_image("dashboard_images/patch_prot_stats_final.png", width=700, padding=30)
    st.markdown("""
    The model doesn’t just look at the whole image. It focuses on important **image sections** (indacted with a yellow box) — like a wing, tail, or head.

    These image sections are compared to **reference features**: small image pieces the model has **learned to recognize** during training.

    - A **reference feature** might come from any bird, not just one class.
    - It represents a **recurring part** that helps the model decide (like a yellow beak, or speckled wing).
    - A reference feature can include pieces from **different birds**, as long as they share similar features.

    For each image section and reference feature pair, you’ll see:
    - **Similarity**: How close your image section is to the reference.
    - **Importance**: How often the predicted bird occurs in the reference (i.e. if 8 out of 10 birds in the reference are a Baltimore Oriole (the predicted class in this case), the importance will be High).
    - **Contribution**: How much this one image section affects the overall prediction score (calculated as the similarity multiplied with the importance).

    These numbers are shown along with a **green contribution bar** — longer bars mean bigger influence.
    """)

    st.markdown("> ##### 📚 Example Interpretation")
    show_centered_image("dashboard_images/patch_prot_stats_final.png", width=700, padding=30)
    st.markdown("""
    > **Image Section #1** (highlighted on the left) matches **Reference Feature #439** with high similarity (**92.7%**). This reference feature is considered to have **High** importance (i.e. the Baltimore Oriole occurs a lot in the reference). As a result, it contributes **12.9 points** (i.e. similarity x importance), which is **39.8%** of the total prediction strength of **32.5**. This tells us that this single match plays a **major role** in why the model predicts this bird class.
    """)
    
    st.markdown("---")
    st.markdown("#### 4️⃣ Can I change the model’s reasoning?")
    show_centered_image("dashboard_images/importance_final.png", width=500, padding=30)
    st.markdown("""
    Yes! Under each image section-reference explanation, you can **change how important the predicted bird** (i.e. Baltimore Oriole) **is in the reference feature**.

    Why this matters:
    - Sometimes the model gives **too much importance** to a bird that does not occur that often in the reference feature or vice versa.
    - By changing this importance (Very Low → Very High), you can directly improve the model's predictions.

    You can:
    - 🔁 **Reset** back to original importance
    - 🚀 **Apply** a new importance level and see how the explanation changes

    ⚠️ *Note*: These changes can potentially change the **top 3 predictions**. For example, by reducing the importance of a reference feature of the first predicted class, its total prediction confidence will shrink, and the second predicted class could now have a bigger prediction confidence than the first predicted class. The dashboard will then automatically swap the spots of these 2 predictions.    """)

    st.markdown("---")
    st.markdown("##### 🧠 When should I update importance?")
    st.markdown("""
    You might want to adjust importance when:

    - The model gives **too much credit** to the predicted bird in the reference feature (e.g. importance is high, but predicted bird only occurs in 2 out of 10 images in reference feature)
    - The model gives **too little credit** to the predicted bird in the reference feature (e.g. importance is low, but predicted bird occurs in 9 out of 10 images in reference feature)

    This tool is not just about accepting the model’s answer — it’s about asking:  
    **"Why did it think that?"** and **"What if it had seen it differently?"**
    """)

    st.markdown("---")
    st.markdown("##### 🔄 Can the top-3 predictions change?")
    st.markdown("""
    Yes — updating the importance of reference features doesn't only change the **contribution bars**,  
    it can also shift which classes are ranked highest in the prediction list.

    - If you lower the importance of the predicted bird in a reference feature that's heavily contributing to a class, its **confidence score will drop**.
        - And the **importances of the other birds** in the reference feature will increase (and thus their confidence score will increase).
    - After this change, the other (unchanged) predicted birds can potentially **overtake** the predicted bird whose confidence score dropped
        - Also **new bird predictions** can enter the top 3 (e.g. predictions that first were placed at 4th or 5th in the ranking)
    - This lets you simulate **alternate model behavior** and test "what if" scenarios.

    For example:
    > Let's say we have top 3 predictions of **Class A** with **39%** confidence, **Class B** with **25%**, and **Class C** with **8%** confidence
    
    > Let's say there is a **reference feature 123** in which Class A has a **High** importance, Class B has a **Low** importance, and Class D has a **Very Low** importance
    
    > If you reduce the importance of Class A in that reference feature 123 **from High to Very Low**, the total confidence score might drop **from 39% to 27%**
    
    > Additionally, the drop in importance of Class A in reference feature 123 is **compensated** with an increase in importance for **Class B and D** in that reference feature, to **High and Moderate** respectively
    
    > Therefore, Class B's total confidence score may increase **from 25% to 33%**, and therefore Class B **overtakes** the number 1 prediction spot of Class A
    
    > Similarly, Class D, which was not in the top 3 initially, might see his total confidence score increase **from 4% to 10%**, therefore **overtaking** Class C's top 3 spot
    
    > Thus, by one simple importance update, the top 3 predictions switched from: 
    
    > - **1. Class A** (39%)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2. Class B** (25%)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**3. Class C** (8%)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to: 
    > - **1. Class B** (33%)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2. Class A** (27%)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**3. Class D** (10%)

    These changes help you explore not just **"Why did the model think this?"**,  
    but also **"What would it think instead?"**
    """)


    st.markdown("---")
    st.markdown("## 💬 Summary")
    st.markdown("""
    - ✅ Use the carousels and image sections to see what the model "sees".
    - 🔍 Compare your photo to reference features.
    - 🎚️ Adjust importance to test model understanding and improve the model.
    - 🧩 Remember: Reference features can come from different birds — it's about shared patterns, not one bird per feature.

    This dashboard helps you understand **how AI makes decisions**, not just what the decisions are.
    """)
