"""Lab 05 bonus: live HOG-based steel surface inspection with Streamlit."""

from pathlib import Path
from threading import Lock
import time

import av
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog
from streamlit_webrtc import webrtc_streamer


MODEL_PATH = Path(__file__).resolve().parent / "inspection_model.joblib"


@st.cache_resource
def load_model(path: str):
    # Joblib files can execute code while loading. Use only the model from
    # the Lab 05 notebook that you ran yourself.
    bundle = joblib.load(path)
    expected = {"model", "classes", "image_size", "hog_cell_size", "hog_orientations"}
    if not isinstance(bundle, dict) or not expected.issubset(bundle):
        raise ValueError("This is not a Lab 05 inspection model.")
    return bundle


def classify(image: Image.Image, bundle: dict) -> tuple[str, float, pd.DataFrame]:
    # This must match preprocessing and HOG parameters in the notebook.
    gray = np.asarray(
        image.convert("L").resize(tuple(bundle["image_size"]), Image.Resampling.BILINEAR)
    )
    features = hog(
        gray,
        orientations=int(bundle["hog_orientations"]),
        pixels_per_cell=(int(bundle["hog_cell_size"]),) * 2,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)[None, :]
    model = bundle["model"]
    probabilities = model.predict_proba(features)[0]
    order = np.argsort(probabilities)[::-1]
    scores = pd.DataFrame({
        "Defect category": np.asarray(model.classes_)[order],
        "Estimated probability": probabilities[order],
    })
    return str(scores.iloc[0, 0]), float(scores.iloc[0, 1]), scores


from skimage import filters, measure, morphology

def detect_defects(image: Image.Image, bundle: dict, stride: int, threshold: float) -> list:
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    patch_size = bundle["image_size"][0]
    
    import scipy.ndimage as ndi
    
    # 1. Edge-based Region Proposal (Find the defect areas first)
    # Use Gaussian Blur valley detection (100x faster than morphological black tophat)
    # Convert to float for subtraction
    gray_float = gray.astype(float)
    blurred_heavy = filters.gaussian(gray_float, sigma=15.0) * 255
    tophat_approx = blurred_heavy - gray_float
    
    # Also find sharp edges (for scratches)
    blurred = filters.gaussian(gray, sigma=2.0)
    edges = filters.sobel(blurred)
    
    # Combine both signals
    try:
        mask1 = tophat_approx > filters.threshold_otsu(tophat_approx)
        mask2 = edges > filters.threshold_otsu(edges)
        mask = mask1 | mask2
    except Exception:
        mask = (tophat_approx > np.percentile(tophat_approx, 95)) | (edges > np.percentile(edges, 95))
        
    # Dilate strongly to merge the entire crack into a single object
    # Use scipy binary_dilation with iterations (100x faster than skimage square(35))
    mask = ndi.binary_dilation(mask, iterations=17)
    mask = morphology.remove_small_objects(mask, min_size=1500)
    
    label_image = measure.label(mask)
    regions = measure.regionprops(label_image)
    
    patches = []
    coords = []
    
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        # Add padding
        pad = 10
        minr, minc = max(0, minr - pad), max(0, minc - pad)
        maxr, maxc = min(h, maxr + pad), min(w, maxc + pad)
        
        # Only process regions that are reasonably sized
        if (maxr - minr) < 10 or (maxc - minc) < 10:
            continue
            
        patch = gray[minr:maxr, minc:maxc]
        # Resize to 80x80 (model input size)
        import PIL.Image
        patch_img = PIL.Image.fromarray(patch).resize((patch_size, patch_size), PIL.Image.Resampling.BILINEAR)
        patches.append(np.asarray(patch_img))
        coords.append((minc, minr, maxc, maxr))
            
    if not patches:
        return []
        
    orientations = int(bundle["hog_orientations"])
    pixels_per_cell = (int(bundle["hog_cell_size"]),) * 2
    
    features_list = []
    for p in patches:
        f = hog(
            p, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2), block_norm="L2-Hys",
            transform_sqrt=True, feature_vector=True,
        )
        features_list.append(f)
        
    features_arr = np.asarray(features_list, dtype=np.float32)
    model = bundle["model"]
    probs = model.predict_proba(features_arr)
    classes = np.asarray(model.classes_)
    
    results = []
    for i in range(len(patches)):
        max_prob = np.max(probs[i])
        class_idx = np.argmax(probs[i])
        label = classes[class_idx]
        
        if label != "normal" and max_prob >= threshold:
            area = (coords[i][2] - coords[i][0]) * (coords[i][3] - coords[i][1])
            results.append((coords[i], label, max_prob, area))
            
    return results


def status_for(label: str) -> tuple[str, str]:
    if label == "normal":
        return "NON-DEFECTIVE", "ACCEPT PRODUCT"
    return f"DEFECTIVE: {label.upper()}", "REJECT PRODUCT"


def overlay_font():
    for font_path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, 22)
    return ImageFont.load_default()


st.set_page_config(page_title="Lab 05 Steel Inspection", page_icon="🔎", layout="centered")
st.title("Steel surface inspection")
st.caption("Lab 05 bonus · HOG features and the classifier trained in your notebook")

if not MODEL_PATH.is_file():
    st.error("The trained model is missing. Run the Lab 05 notebook, download Lab05_Results.zip, "
             "and place app.py in its lab05_results folder beside inspection_model.joblib.")
    st.stop()

try:
    bundle = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

st.write(f"Trained model: **{bundle.get('model_name', 'Lab 05 classifier')}**")
if "normal" not in bundle["classes"]:
    st.warning("This training dataset has only defect classes. The app can name a likely defect, "
               "but it cannot establish that a normal product is safe to accept.")

st.sidebar.header("Detection Settings")
enable_detection = st.sidebar.checkbox("Enable Bounding Boxes", value=True)
st.sidebar.caption("Uses region proposals + HOG to locate defects.")
detection_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.50, step=0.05)

st.sidebar.markdown("""
**Bounding Box Colors:**
- 🟢 **Lime Green:** Small defects
- 🟠 **Orange:** Medium defects
- 🔴 **Red-Orange:** Large critical defects
""")

photo_tab, live_tab = st.tabs(["Upload a photo", "Live webcam"])

with photo_tab:
    uploaded = st.file_uploader("Choose a steel surface image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            label, confidence, scores = classify(image, bundle)
            if enable_detection:
                with st.spinner("Finding defects..."):
                    detections = detect_defects(image, bundle, 0, detection_threshold)
            else:
                detections = []
        except Exception as exc:
            st.error(f"Could not inspect the image: {exc}")
        else:
            if enable_detection:
                draw_img = image.copy()
                draw = ImageDraw.Draw(draw_img)
                font = overlay_font()
                for (x1, y1, x2, y2), d_label, d_conf, d_area in detections:
                    # Multi-color bounding boxes based on size
                    if d_area < 5000:
                        color = (50, 205, 50)  # Lime Green for Small
                        size_prefix = "Small"
                    elif d_area < 25000:
                        color = (255, 165, 0)  # Orange for Medium
                        size_prefix = "Medium"
                    else:
                        color = (255, 69, 0)   # Red-Orange for Large
                        size_prefix = "Large"
                        
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    text = f"{size_prefix} {d_label} ({d_conf:.0%})"
                    
                    # Draw a semi-transparent text background
                    bbox = font.getbbox(text)
                    draw.rectangle([x1, y1 - 22, x1 + bbox[2], y1], fill=color)
                    draw.text((x1 + 2, y1 - 22), text, font=font, fill=(0, 0, 0))
                    
                st.image(draw_img, caption="Detected Defects", use_container_width=True)
            else:
                st.image(image, caption="Selected image", use_container_width=True)
            prediction, action = status_for(label)
            st.subheader("PRODUCT INSPECTION RESULT")
            st.metric("Prediction", prediction)
            st.metric("Estimated confidence", f"{confidence:.1%}")
            if action == "ACCEPT PRODUCT":
                st.success(action)
            else:
                st.error(action)
            st.bar_chart(scores.set_index("Defect category"))
            st.dataframe(scores.style.format({"Estimated probability": "{:.1%}"}),
                         hide_index=True, use_container_width=True)

with live_tab:
    st.write("Click **START** and allow browser camera access. Hold a steel surface in view. "
             "The prediction is drawn on the live video about once per second.")
    font = overlay_font()
    state_lock = Lock()
    state = {"last_time": 0.0, "prediction": "Waiting for a frame", "confidence": 0.0,
             "action": "", "detections": []}

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
        now = time.monotonic()
        with state_lock:
            if now - state["last_time"] >= 0.8:
                try:
                    label, confidence, _ = classify(image, bundle)
                    state["prediction"], state["action"] = status_for(label)
                    state["confidence"] = confidence
                    if enable_detection:
                        state["detections"] = detect_defects(image, bundle, 0, detection_threshold)
                    else:
                        state["detections"] = []
                except Exception:
                    state["prediction"] = "Unable to inspect frame"
                    state["action"] = ""
                    state["detections"] = []
                state["last_time"] = now
            prediction = state["prediction"]
            confidence = state["confidence"]
            action = state["action"]
            detections = state["detections"]

        draw = ImageDraw.Draw(image)
        
        for (x1, y1, x2, y2), d_label, d_conf, d_area in detections:
                    if d_area < 5000:
                        color = (50, 205, 50)
                        size_prefix = "Small"
                    elif d_area < 25000:
                        color = (255, 165, 0)
                        size_prefix = "Medium"
                    else:
                        color = (255, 69, 0)
                        size_prefix = "Large"
                        
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    text = f"{size_prefix} {d_label} ({d_conf:.0%})"
                    bbox = font.getbbox(text)
                    draw.rectangle([x1, y1 - 22, x1 + bbox[2], y1], fill=color)
                    draw.text((x1 + 2, y1 - 22), text, font=font, fill=(0, 0, 0))
            
        draw.rectangle((0, 0, image.width, 76), fill=(13, 26, 43))
        draw.text((12, 7), prediction, font=font, fill=(255, 255, 255))
        if action:
            draw.text((12, 40), f"{action} | confidence {confidence:.0%}",
                      font=font, fill=(250, 205, 94))
        return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")

    webrtc_streamer(
        key="lab05-live-inspection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

st.caption("Classifies whole images only. Use a separate set of real normal products "
           "before considering automatic acceptance in production.")
