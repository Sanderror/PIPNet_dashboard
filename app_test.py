# --- IMPORTS ---
import streamlit as st
import shutil
import subprocess
from pathlib import Path
import base64
import re
import streamlit.components.v1 as components
from PIL import Image
import torch

# --- CONFIGURATION ---
### DOES NOT DO ANYTHING. NEED TO CONFIGURE COLORS IN THE DASHBOARD INTERFACE ITSELF (SETTINGS) OR IN YOUR ENVIRONMENT > /streamlit/config.toml

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

# --- SESSION STATE ---
st.session_state.setdefault("current_tab", "Upload")
st.session_state.setdefault("predictions", None)
st.session_state.setdefault("uploaded", False)

# --- UTILITY FUNCTIONS ---
def clear_folder(folder):
    if folder.exists():
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
            else:
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

def get_all_class_images(class_name):
    for folder in CLASS_IMAGES_DIR.iterdir():
        if folder.is_dir() and class_name in folder.name:
            return list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    return []

def extract_patch_info(filename):
    try:
        return {
            "mul": float(re.search(r"mul([0-9.]+)", filename).group(1)) if re.search(r"mul([0-9.]+)", filename) else None,
            "sim": float(re.search(r"sim([0-9.]+)", filename).group(1)) if re.search(r"sim([0-9.]+)", filename) else None,
            "w": float(re.search(r"w([0-9.]+)", filename).group(1)) if re.search(r"w([0-9.]+)", filename) else None,
        }
    except Exception:
        return {"mul": None, "sim": None, "w": None}

def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

def show_image_carousel(image_paths, height=400):
    if not image_paths:
        st.warning("No images to display.")
        return

    slides_html = "".join(f"<div class='slide'><img src='{image_to_base64(img)}' style='width:100%'></div>" for img in image_paths)

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
    <div id="slideIndexLabel" style="text-align: center; font-size: 12px; color: gray; margin-top: 8px;"></div>
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

    components.html(carousel_html, height=height+100)

def get_sorted_rect_images(image_folder, class_name):
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
    parts = patch_filename.split("_")
    for part in parts:
        if part.startswith("p") and part[1:].isdigit():
            return f"prototype_{part[1:]}"
    return None

def get_prototype_images(prototype_id):
    folder = PROTOTYPE_DIR / prototype_id
    if folder.exists() and folder.is_dir():
        return sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    return []

def show_centered_image(path, width=400, padding=20):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="text-align: center; padding-top: {padding}px; padding-bottom: {padding}px;">
        <img src="data:image/png;base64,{data}" width="{width}">
    </div>
    """, unsafe_allow_html=True)

# --- UI ---
st.markdown("<h1 style='text-align: center;'>🧠 PIPNet Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

tabs = ["About", "Upload", "Predictions", "Explanation"]
selected_tab = st.radio("", tabs, index=tabs.index(st.session_state.current_tab), horizontal=True)
st.markdown("---")

# --- TAB CONTENT ---
if selected_tab == "About":
    st.header("📖 About")
    st.markdown("""
    Welcome to the PIPNet Dashboard!  
    - Upload bird images.
    - Predict and explore how the model makes decisions.
    - See patch-to-prototype explanations.
    """)

elif selected_tab == "Upload":
    st.header("📤 Upload your images")
    uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        save_uploaded_files(uploaded_files)
        st.session_state.uploaded = True
        st.success(f"{len(uploaded_files)} image(s) uploaded successfully!")
        for img in uploaded_files:
            st.image(img, caption=img.name, width=250)

    if st.button("Predict"):
        with st.spinner("Running model..."):
            run_prediction_script()
        st.session_state.predictions = parse_predictions()
        st.session_state.current_tab = "Predictions"
        st.rerun()

elif selected_tab == "Predictions":

    if not st.session_state.predictions:
        st.info("Please upload images and predict first.")
    else:
        predictions = st.session_state.predictions

        # Prepare image options
        image_options = list(predictions.keys())
        st.header("Select an image to view its predictions")
        selected_image_id = st.selectbox(f"Select one of your {len(image_options)} uploaded images in the dropdown menu below:", image_options)

        # Obtain the path to the selected image
        preds = predictions[selected_image_id]
        matches = list(USER_IMAGES_DIR.glob(f"{selected_image_id}.*"))
        image_path = matches[0] if matches else None

        st.header(f"🔎 Top 3 Predictions for Image {selected_image_id}")
        st.image(image_path, caption=f"Uploaded image {selected_image_id}", width=400)
        st.markdown("---")

        # For each uploaded image, we show the top 3 predicted classes with the explanations
        for prediction_idx, (cls, score) in enumerate(preds, start=1):
            # Obtain the bird name
            display_cls = " ".join(cls.split(".")[-1].split("_"))
            st.markdown(f"### {prediction_idx}. {display_cls} — Score: {score:.3f}")

            # Create a slideshow of images of the predicted class
            class_images = get_all_class_images(cls)
            if class_images:
                show_image_carousel(class_images)
                rects = get_sorted_rect_images(selected_image_id, cls)
            else:
                st.warning(f"No images found for {display_cls}")

            # Create a section where the patch in the image is shown, then the related prototype with a slideshow, and then stats on the simalirty and scores
            if rects:
                header_patch, header_proto, header_stats = st.columns([1, 1, 1])
                with header_patch:
                    st.markdown("##### 🟨 Patch")
                with header_proto:
                    st.markdown("##### 🧬 Prototype")
                with header_stats:
                    st.markdown("##### 📊 Stats")
                st.markdown("---")

                rects_total_score = sum(
                    extract_patch_info(r.name)["mul"] for r in rects if extract_patch_info(r.name)["mul"] is not None
                )

                # Create a separate section for each detected prototype in the image
                for idx, img_path in enumerate(rects, start=1):
                    prototype_id = get_prototype_folder(img_path.name)
                    prototype_num = prototype_id.replace("prototype_", "") if prototype_id else "?"
                    prototype_imgs = get_prototype_images(prototype_id)
                    info = extract_patch_info(img_path.name)
                    fixed_height = 160

                    col_patch, col_proto, col_stats = st.columns(3)
                    # The patch
                    with col_patch:
                        st.image(str(img_path), caption=f"#{idx}", width=fixed_height)
                    # The prototype
                    with col_proto:
                        if prototype_imgs:
                            show_image_carousel(prototype_imgs, height=fixed_height)
                        else:
                            st.warning("No prototype found")
                    # The stats
                    with col_stats:
                        if all(v is not None for v in info.values()):
                            mul = info["mul"]
                            percent = (mul / rects_total_score) * 100 if rects_total_score else 0

                            st.markdown(f"""
                            ##### **Prototype {prototype_num}**  
                            • **Similarity**: `{info['sim']:.3f}`  
                            • **Weight**: `{info['w']:.3f}`  
                            • **Mul**: `{info['mul']:.3f}`  
                            • **Contribution**: `{percent:.1f}%`
                            <div style="background-color: #e0e0e0; width: 100%; height: 20px; border-radius: 5px; overflow: hidden; margin-top: 5px;">
                                <div style="background-color: #4CAF50; width: {percent}%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 12px; color: white; font-weight: bold;">
                                    {info['mul']:.1f}
                                </div>
                            </div>
                            <div style="text-align: right; font-size: 12px; color: gray; margin-top: 4px;">
                                Total Class Score: {rects_total_score:.1f}
                            </div>
                            """, unsafe_allow_html=True)

                    ### SECTION FOR ADJUSTING THE WEIGHTS
                    # Sorry, I know this is a shit-hole right now. I will clean it up later :) at least it works

                    # Obtain the weights from a file in which they are stored
                    net_weights = torch.load(NET_WEIGHTS_DIR)
                    net_weights = net_weights.detach().clone()

                    # When indexing the weights, the index for class 1: Black Footed Albatross, is actually 0. Therefore the -1
                    cls_index = int(cls.split(".")[0]) - 1

                    # Obtain the weights of every class for the selected prototype
                    prototype_weights = net_weights[:, int(prototype_num)]

                    # Obtain the weight of the current predicted class for this prototype (this should be the same as the info['w'] value)
                    class_prototype_weight = float(net_weights[cls_index, int(prototype_num)])

                    # Obtain the sum of all weights for the this prototype
                    total_prototype_weight = float(sum(prototype_weights))

                    # Sort the weights from highest to lowest for this prototype
                    highest_weight_classes = torch.argsort(prototype_weights, descending=True)

                    st.markdown(f"""
                    ##### **Adjust weights Class {display_cls} in Prototype {prototype_num}**
                    ######      ***Help us improve the model by adjusting the weights!*** \n
                    In **Prototype {prototype_num}**, the predicted class **{display_cls}** has a **weight of {round(class_prototype_weight,3)}** 
                    ({round(class_prototype_weight/total_prototype_weight*100, 1)}% of total weight). \n
                    The total weight for this prototype is **{round(total_prototype_weight,3)}** and is divided over the **following classes**:
                    """, unsafe_allow_html=True)

                    # Loop over all classes in this prototype, and print out their weight if it is bigger than 0
                    prototype_images = dict()
                    for weight_idx in highest_weight_classes:
                        # Get the weight for a specific class in this prototype
                        class_weight = float(prototype_weights[weight_idx].item())
                        if class_weight > 0:
                            # To get the IMAGE ID (the class e.g. 001 for Black footed albatross) we should know do +1 again, because image ids start at 1 not at 0
                            img_idx = weight_idx + 1
                            # Transform the image id to the correct format to obtain the bird name
                            if img_idx < 10:
                                img_idx = f"00{int(img_idx)}"
                            elif img_idx >= 10 and img_idx < 100:
                                img_idx = f"0{int(img_idx)}"
                            else:
                                img_idx = str(int(img_idx))
                            # Obtain the bird name using the image id
                            img_dir = [d.name for d in CLASS_IMAGES_DIR.iterdir() if d.name.startswith(img_idx)][0]
                            bird_name = " ".join(img_dir.split(".")[1].split("_"))
                            prototype_images[img_dir] = class_weight
                            st.markdown(f"  • **{bird_name}**: {round(class_weight,3)} ({round(class_weight/total_prototype_weight*100, 1)}% of total weight)")

                    # Section to update the weights of the predicted class in this prototype
                    st.markdown(f"""
                    ###### Would you like to update the weight of Class {display_cls} in Prototype {prototype_num}?
                    """)

                    # A whole lot of keys needed for different streamlit functionalities
                    form_key = f"form_{selected_image_id}_{cls}_{prototype_num}"
                    slider_key = f"slider_{selected_image_id}_{cls}_{prototype_num}"
                    internal_slider_key = slider_key + "_internal"
                    reset_button_key = f"reset_{selected_image_id}_{cls}_{prototype_num}"
                    submit_button_key = f"submit_{selected_image_id}_{cls}_{prototype_num}"
                    show_followup_key = f"show_followup_{selected_image_id}_{cls}_{prototype_num}"
                    adjustment_submit_key = f"submit_adjustment_{selected_image_id}_{cls}_{prototype_num}"
                    adjustment_form_key = f"adjustment_form_{selected_image_id}_{cls}_{prototype_num}"

                    # Initializes the values for weight adjusting sliders
                    if slider_key not in st.session_state:
                        st.session_state[slider_key] = class_prototype_weight
                    if internal_slider_key not in st.session_state:
                        st.session_state[internal_slider_key] = class_prototype_weight
                    if show_followup_key not in st.session_state:
                        st.session_state[show_followup_key] = False

                    # If the reset button has been pressed. The state has to be reset to the original class weight
                    if f"reset_{form_key}" in st.session_state and st.session_state[f"reset_{form_key}"]:
                        st.session_state[internal_slider_key] = class_prototype_weight
                        st.session_state[f"reset_{form_key}"] = False  # Reset the trigger flag

                    # Section for a slider to adjust the weights
                    with st.form(key=form_key):
                        new_value = st.slider(
                            f"Adjust the weight of class '{display_cls}' in prototype {prototype_num}",
                            min_value=0.0,
                            max_value=total_prototype_weight,
                            value=st.session_state[internal_slider_key], # initialized at the current weight
                            step=0.001,
                            format="%0.3f",
                            key=internal_slider_key
                        )

                        # Create two buttons: one to reset the slider to its original weight, one to submit the current weight of the slider
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            reset_clicked = st.form_submit_button("🔄 Reset Weight")
                        with col2:
                            submit_clicked = st.form_submit_button("🚀 Submit weights")

                    # To handle the reset button
                    if reset_clicked:
                        st.session_state[f"reset_{form_key}"] = True
                        st.session_state[show_followup_key] = False
                        st.rerun()

                    # Save new weight value (for the class in the prototype) to a state
                    st.session_state[slider_key] = new_value

                    # Detect weight change direction
                    weight_change = new_value - class_prototype_weight
                    change_direction = "increase" if round(weight_change,3) > 0 else "decrease" if round(weight_change,3) < 0 else "no_change"
                    negative_direction = 'Decrease' if change_direction == 'increase' else 'Increase'

                    # If the submit button has been clicked, and the weight of the slider has been changed, then a next section will pop up
                    if submit_clicked and change_direction != "no_change":
                        st.session_state[show_followup_key] = True

                    # This is to show a text which says by which margin the weight has been updated
                    if change_direction != "no_change":
                        delta = round(abs(weight_change), 3)
                        percent = round(new_value / total_prototype_weight * 100, 1)
                        color = "green" if weight_change > 0 else "red"
                        arrow = "⬆️" if weight_change > 0 else "⬇️"
                        st.markdown(
                            f"""<span style='color: {color};'>{arrow} Weight {change_direction} by {delta} to: <b>{new_value:.3f}</b> ({percent}% of total weight)</span>""",
                            unsafe_allow_html=True
                        )

                    # This is where the user has to decide HOW he wants to perform the adjusting of weights
                    if st.session_state[show_followup_key]:
                        st.markdown(f"""
                        ### You are going to **{change_direction}** the weight for the class **{display_cls}** in **Prototype {prototype_num}** from \
                        {class_prototype_weight:.3f} to {new_value:.3f}.
                        #### How do you want to deal with this difference?
                        """)

                        # There are 3 options: simply increase total weight, change other classes proportionally, change other classes yourself
                        with st.form(key=adjustment_form_key):
                            user_choice = st.radio(
                                "Choose one of the following options:",
                                options=[
                                    "Do not touch the weights of other classes; just update the total weight (**RECOMMENDED**)",
                                    f"{negative_direction} the weights of the other classes proportionately",
                                    f"{negative_direction} the weights of the other classes yourself",
                                ],
                                key=f"adjustment_choice_{selected_image_id}_{cls}_{prototype_num}_radio"
                            )
                            # Submit the choice of adjustment method with a button
                            submit_adjustment = st.form_submit_button("Perform Adjustment Method")

                        # If the submit button has been clicked, the weights will be updated for the prototype and
                        # the affected classes according to the adjustment method selected
                        if submit_adjustment:
                            if user_choice == 'Do not touch the weights of other classes; just update the total weight (**RECOMMENDED**)':
                                net_weights[cls_index, int(prototype_num)] = new_value
                                # The new weights will be saved to the same directory, so therefore it is also instantly used in the dashboard
                                torch.save(net_weights, NET_WEIGHTS_DIR)
                                st.session_state[show_followup_key] = False
                                st.session_state["show_success_message"] = True
                                st.rerun()
                            elif user_choice == f"{negative_direction} the weights of the other classes proportionately":
                                # To obtain the other classes for this prototype
                                remaining_imgs = [(img, scr) for img, scr in prototype_images.items() if img != cls]
                                remaining_score = sum([key[1] for key in remaining_imgs])
                                # If there are no other classes, then this option is not possible
                                if remaining_score == 0:
                                    st.error("Cannot redistribute: No other classes have weights for this prototype. Try a different adjustment method.")
                                else:
                                    # All other classes will be updated proportionally using ratios to the total
                                    # E.g. if the weights are [5,5,10] and one gets increased from 5 to 10, then the other 5 will be decreased
                                    # with 5 * 5/(5+10) = 5/3 = 1.667 (so new score is 3.333) and the 10 will be decreased with 5 * 10/(5+10) = 10/3 = 3.3333
                                    # (so new score will be 6.667). So then we have 3.333+ 5 + 6.667 = 20, so still the same total weight
                                    dict_new_weights = dict()
                                    for key in remaining_imgs:
                                        img, scr = key[0], key[1]
                                        ratio = scr / remaining_score
                                        temp_index = int(img.split('.')[0]) - 1
                                        temp_weight = net_weights[temp_index, int(prototype_num)] - weight_change * ratio
                                        # Update the other classes
                                        net_weights[temp_index, int(prototype_num)] = temp_weight
                                        dict_new_weights[img] = temp_weight
                                    # Store this for plotting
                                    dict_new_weights[cls] = new_value
                                    st.session_state['post_update_message'] = dict_new_weights
                                    # Update the main class
                                    net_weights[cls_index, int(prototype_num)] = new_value
                                    st.markdown(f"Class {display_cls} received a new weight of {new_value}")
                                    # Save it to the same directory
                                    torch.save(net_weights, NET_WEIGHTS_DIR)
                                    st.session_state[show_followup_key] = False
                                    st.session_state["show_success_message"] = True
                                    st.rerun()
                            else:
                                st.markdown("Not implemented yet...")

                    # Plot success message, to show the weights have been updated well
                    if st.session_state.get("show_success_message", False):
                        st.success("🎉 Weights have been updated successfully!")
                        st.session_state["show_success_message"] = False
                    if "post_update_message" in st.session_state:
                        for img_updated, weight_updated in st.session_state['post_update_message'].items():
                            st.markdown(f"Class {img_updated} received a new weight of {weight_updated}")
                        del st.session_state['post_update_message']

                    st.markdown("---")
            else:
                st.markdown("Something went wrong with finding the prototypes")
            prediction_idx += 1

elif selected_tab == "Explanation":
    st.header("❓ How to Interpret Predictions")

    st.markdown("### **1. Class Name and Score**")
    show_centered_image("dashboard_images/name_score.png", width=275, padding=30)
    st.markdown("Each prediction shows the **predicted class** and its **confidence score**. A higher score means the model is more confident about that class.")

    st.markdown("### **2. Class Carousel**")
    show_centered_image("dashboard_images/class_carousel.png", width=350, padding=30)
    st.markdown("A small gallery of typical **images from the predicted class**. Helps you visually compare your uploaded image to known examples.")

    st.markdown("### **3. Patch, Prototype, and Stats**")
    show_centered_image("dashboard_images/patch_prot_stats.png", width=700, padding=30)
    st.markdown("""
    Each prediction is built from **patches** of the uploaded image. Each patch:
    - Is compared to a **learned prototype** (e.g., bird heads, wings, tails).
    - Computes a **similarity score** (how close the patch looks to the prototype).
    - Applies a **weight** (how important the prototype is for recognizing that class). All patches' weighted similarities are summed to calculate the final class score.
    - A **contribution bar** shows how much a patch's prototype contributed to the overall score. Longer bars = bigger impact.
    """)

    st.markdown("---")
    st.header("📚 Example Interpretation")
    show_centered_image("dashboard_images/patch_prot_stats.png", width=700, padding=30)
    st.markdown("> ***Patch #1** is highly similar (**92.7%**) to **Prototype #439**, contributing **12.9** points out of a total **32.5 class score**. This prototype (e.g., a bird head) is crucial for recognizing the 'Baltimore Oriole'.*")

    st.markdown("---")
    st.header("💡 Why It's Useful")
    st.markdown("- See **which parts** of the image matter most.\n- Understand **model reasoning**, not just the final answer.\n- Find cases where the model is **confused** (wrong prototype matches).")
