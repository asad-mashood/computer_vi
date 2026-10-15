"""
LandmarkAI: SIFT-Based Landmark Recognition & Geometric Localization Web Application
Author: Muhammad Asad Mashood (FA23-BAI-039)
Course: Computer Vision Lab 06
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LandmarkAI – SIFT Recognition System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #0d1322 0%, #0b0f19 90%);
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.7);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
}

.hero-header {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.12) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(2, 132, 199, 0.2);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-subtitle {
    color: #94a3b8;
    margin: 6px 0 0 0;
    font-size: 0.95rem;
}

.badge-rec {
    background: linear-gradient(90deg, #059669, #10b981);
    color: #ffffff;
    font-weight: 700;
    padding: 10px 24px;
    border-radius: 30px;
    font-size: 1.15rem;
    display: inline-block;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
}

.badge-rej {
    background: linear-gradient(90deg, #dc2626, #ef4444);
    color: #ffffff;
    font-weight: 700;
    padding: 10px 24px;
    border-radius: 30px;
    font-size: 1.15rem;
    display: inline-block;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
}

.stat-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    backdrop-filter: blur(8px);
}

.stat-val {
    font-size: 1.7rem;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 2px;
}

.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. RESOLVE DATABASE PATH
# ─────────────────────────────────────────────────────────────────────────────
base_script_dir = os.path.dirname(os.path.abspath(__file__))
candidates = [
    os.path.join(base_script_dir, "data", "landmarks"),
    os.path.join(base_script_dir, "lab06", "Landmark_Recognition_SIFT", "data", "landmarks"),
    r"D:\ComputerVision-LABS-Asad\lab06\Landmark_Recognition_SIFT\data\landmarks",
    r"D:\Comp Vision Lab 6 Asad\data\landmarks",
    r"D:\Cv lab 6_staging\landmarks"
]

DATA_DIR = None
for c in candidates:
    if os.path.exists(c) and len(os.listdir(c)) >= 5:
        DATA_DIR = c
        break

if not DATA_DIR:
    st.error("Error: Could not locate Landmark Database folder. Please verify repository structure.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 3. CACHED SIFT DATABASE INGESTION & FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_index_database(data_path):
    sift = cv2.SIFT_create()
    database = {}
    total_views = 0
    total_keypoints = 0

    categories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d != 'unknown_landmark']
    for cat in sorted(categories):
        p = os.path.join(data_path, cat)
        files = sorted(glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.png")))
        database[cat] = []
        for f in files:
            bgr = cv2.imread(f)
            if bgr is not None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                kps, descs = sift.detectAndCompute(gray, None)
                database[cat].append({
                    'file': os.path.basename(f),
                    'path': f,
                    'bgr': bgr,
                    'gray': gray,
                    'kps': kps,
                    'descs': descs
                })
                total_views += 1
                total_keypoints += len(kps)
    return database, total_views, total_keypoints

with st.spinner("Initializing SIFT detector and caching database features..."):
    db, num_views, total_kps = load_and_index_database(DATA_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR CONFIGURATION & TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏛️ LandmarkAI Studio")
    st.markdown("**Student:** Muhammad Asad Mashood  
**ID:** `FA23-BAI-039`  
**Course:** Computer Vision (Lab 06)")
    st.divider()

    st.markdown("#### ⚙️ SIFT Pipeline Controls")
    ratio_thresh = st.slider("Lowe's Ratio Test (τ)", min_value=0.50, max_value=0.95, value=0.75, step=0.05,
                             help="Filters ambiguous matches: d1 < tau * d2")
    min_matches = st.slider("Rejection Gate (Min Matches)", min_value=5, max_value=50, value=20, step=1,
                            help="Threshold for accepting query as known landmark")
    min_inliers = st.slider("RANSAC Min Inliers", min_value=5, max_value=40, value=12, step=1,
                            help="Required inlier count for geometric pose localization")
    max_kps_draw = st.slider("Max Keypoints to Render", min_value=100, max_value=1500, value=400, step=50)

    st.divider()
    st.markdown("#### 📊 Database Telemetry")
    st.markdown(f"- **Categories:** {len(db)} world landmarks")
    st.markdown(f"- **Reference Views:** {num_views} photographs")
    st.markdown(f"- **Indexed Descriptors:** {total_kps:,} SIFT vectors")

    st.divider()
    st.caption("Powered by OpenCV 5.0 SIFT & Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🏛️ LandmarkAI: SIFT Recognition System</div>
    <div class="hero-subtitle">
        Automated landmark recognition and geometric planar localization using Scale-Invariant Feature Transform,
        Brute-Force KNN matching, Lowe's ratio test, and RANSAC homography.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. QUERY INGESTION CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
c_mode, c_sample = st.columns([1, 2])

with c_mode:
    input_mode = st.radio("Choose Query Source:", ["Pre-configured Test Suite", "Upload Image"], horizontal=True)

query_bgr = None
query_label = ""

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Query Landmark Photograph (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        query_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        query_label = uploaded_file.name
else:
    # Build options
    eiffel_ref = db['eiffel_tower'][0]['bgr']
    h_e, w_e = eiffel_ref.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w_e/2, h_e/2), 45, 0.85)

    sample_dict = {
        "Eiffel Tower – Original (Unperturbed)": eiffel_ref.copy(),
        "Eiffel Tower – Rotated (45° In-Plane)": cv2.warpAffine(eiffel_ref, M_rot, (w_e, h_e), borderMode=cv2.BORDER_REFLECT),
        "Eiffel Tower – Resized (0.5x Scale)": cv2.resize(eiffel_ref, (w_e//2, h_e//2), interpolation=cv2.INTER_AREA),
        "Eiffel Tower – Low Illumination (0.35x Dark)": np.clip(eiffel_ref.astype(np.float32) * 0.35, 0, 255).astype(np.uint8),
        "Eiffel Tower – Alternate Viewpoint": db['eiffel_tower'][2]['bgr'].copy(),
        "Taj Mahal – Alternate View": db['taj_mahal'][1]['bgr'].copy(),
        "Colosseum – Alternate View": db['colosseum'][1]['bgr'].copy(),
        "Great Wall of China – Alternate View": db['great_wall'][1]['bgr'].copy(),
        "Statue of Liberty – Alternate View": db['statue_of_liberty'][1]['bgr'].copy()
    }

    # Add unknown landmark if available
    unknown_p = os.path.join(DATA_DIR, "unknown_landmark", "big_ben_1.jpg")
    if os.path.exists(unknown_p):
        big_ben = cv2.imread(unknown_p)
        if big_ben is not None:
            sample_dict["Non-Database Impostor (Big Ben)"] = big_ben

    with c_sample:
        selected_sample = st.selectbox("Select Test Query Condition:", list(sample_dict.keys()))
        query_bgr = sample_dict[selected_sample]
        query_label = selected_sample

if query_bgr is None:
    st.info("👈 Select a sample from the test suite or upload an image to run recognition.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIFT RECOGNITION PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
t_start = time.time()

sift = cv2.SIFT_create()
gray_q = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY) if len(query_bgr.shape) == 3 else query_bgr
kps_q, des_q = sift.detectAndCompute(gray_q, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
ranked_candidates = []

for cat, items in db.items():
    best_matches = []
    best_ref = None
    for item in items:
        if des_q is None or item['descs'] is None or len(des_q) < 2 or len(item['descs']) < 2:
            continue
        knn = bf.knnMatch(des_q, item['descs'], k=2)
        good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]
        if len(good) > len(best_matches):
            best_matches = good
            best_ref = item

    ranked_candidates.append({
        'landmark': cat,
        'matches_count': len(best_matches),
        'matches': best_matches,
        'best_ref': best_ref
    })

ranked_candidates.sort(key=lambda x: x['matches_count'], reverse=True)
top = ranked_candidates[0] if ranked_candidates else None
second = ranked_candidates[1] if len(ranked_candidates) > 1 else None

is_recognized = False
status_text = ""

if top is None or top['matches_count'] < min_matches:
    is_recognized = False
    status_text = f"Rejected: Match count ({top['matches_count'] if top else 0}) below threshold ({min_matches})"
elif second and second['matches_count'] > 0 and (top['matches_count'] / second['matches_count'] < 1.35) and top['matches_count'] < 35:
    is_recognized = False
    status_text = f"Rejected: Ambiguous match between {top['landmark']} and {second['landmark']}"
else:
    is_recognized = True
    status_text = f"Recognized: {top['landmark'].replace('_', ' ').title()}"

# Geometric Localization via RANSAC
inliers_count = 0
localized_img = query_bgr.copy()
loc_status = "No localization attempted"

if is_recognized and top['best_ref'] and len(top['matches']) >= 4:
    ref_item = top['best_ref']
    src_pts = np.float32([ref_item['kps'][m.trainIdx].pt for m in top['matches']]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_q[m.queryIdx].pt for m in top['matches']]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None and mask is not None:
        inliers_count = int(mask.sum())
        h_r, w_r = ref_item['bgr'].shape[:2]
        h_q, w_q = query_bgr.shape[:2]

        det = np.linalg.det(H[:2, :2])
        corners = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)

        is_convex = cv2.isContourConvex(np.int32(proj))
        area = cv2.contourArea(np.int32(proj))
        q_area = w_q * h_q

        if inliers_count >= min_inliers and det > 0 and is_convex and (0.02 * q_area <= area <= 3.0 * q_area):
            pts = np.int32(proj)
            cv2.polylines(localized_img, [pts], True, (0, 255, 0), 4, cv2.LINE_AA)
            for p in pts:
                cv2.circle(localized_img, tuple(p[0]), 6, (0, 215, 255), -1)
            loc_status = f"Planar pose localized ({inliers_count} inliers)"
        else:
            loc_status = "Geometric validation failed (distortion or low inliers)"
            cv2.putText(localized_img, "GEOMETRIC CHECK FAILED", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
else:
    cv2.putText(localized_img, "REJECTED / UNKNOWN LANDMARK", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

latency_ms = (time.time() - t_start) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# 8. RENDER RESULTS & METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Top Status Banner
if is_recognized:
    lm_title = top['landmark'].replace('_', ' ').title()
    st.markdown(f'<div class="badge-rec">✓ RECOGNIZED: {lm_title} ({top["matches_count"]} Matches, {inliers_count} Inliers)</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="badge-rej">✕ REJECTED: Impostor / Unknown Landmark</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Metric Cards
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{len(kps_q):,}</div><div class="stat-label">Query Keypoints</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{top["matches_count"] if top else 0:,}</div><div class="stat-label">Reliable Matches</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{inliers_count:,}</div><div class="stat-label">RANSAC Inliers</div></div>', unsafe_allow_html=True)
with m4:
    inlier_pct = (inliers_count / top["matches_count"] * 100) if (top and top["matches_count"] > 0) else 0.0
    st.markdown(f'<div class="stat-card"><div class="stat-val">{inlier_pct:.1f}%</div><div class="stat-label">Inlier Ratio</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{latency_ms:.1f}ms</div><div class="stat-label">Total Latency</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9. INTERACTIVE TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_loc, tab_match, tab_kps, tab_rank, tab_db = st.tabs([
    "🎯 Geometric Localization",
    "🔗 Feature Matching (Pairs)",
    "✨ SIFT Keypoints & Orientation",
    "📊 Candidate Ranking",
    "🏛️ Landmark Database"
])

with tab_loc:
    col_q, col_res = st.columns(2)
    with col_q:
        st.markdown("**Original Query Image**")
        st.image(cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_res:
        st.markdown(f"**Pose Localization Overlay** – *{loc_status}*")
        st.image(cv2.cvtColor(localized_img, cv2.COLOR_BGR2RGB), use_container_width=True)

with tab_match:
    if top and top['best_ref'] and len(top['matches']) > 0:
        st.markdown(f"**Top Correspondences vs Best Reference View:** `{top['best_ref']['file']}`")
        matches_vis = cv2.drawMatches(
            query_bgr, kps_q,
            top['best_ref']['bgr'], top['best_ref']['kps'],
            top['matches'][:60], None,
            matchColor=(0, 255, 0) if is_recognized else (0, 0, 255),
            singlePointColor=(180, 180, 180),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        st.image(cv2.cvtColor(matches_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.warning("No feature matches found.")

with tab_kps:
    st.markdown(f"**Difference-of-Gaussians (DoG) Keypoints** ({len(kps_q)} Extrema Detected)")
    kps_vis = cv2.drawKeypoints(
        query_bgr, kps_q[:max_kps_draw], None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    st.image(cv2.cvtColor(kps_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

with tab_rank:
    st.markdown("**Candidate Match Counts vs Database Categories**")
    cat_names = [c['landmark'].replace('_', ' ').title() for c in ranked_candidates]
    cat_scores = [c['matches_count'] for c in ranked_candidates]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.barh(cat_names[::-1], cat_scores[::-1], color='#38bdf8', height=0.55)
    ax.axvline(min_matches, color='#f87171', linestyle='--', linewidth=1.5, label=f'Rejection Gate (T = {min_matches})')

    ax.set_xlabel("Reliable Matches (Lowe's Ratio Passed)", color='#cbd5e1', fontsize=11, fontweight='bold')
    ax.set_title("Candidate Match Scores", color='#f1f5f9', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#cbd5e1', labelsize=10)
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f1f5f9')
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='#475569')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 10, bar.get_y() + bar.get_height()/2.0, f'{int(w)}', ha='left', va='center', color='#38bdf8', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab_db:
    st.markdown("**Registered Multi-View Landmark Reference Database**")
    cols = st.columns(len(db))
    for col, (cat, items) in zip(cols, sorted(db.items())):
        with col:
            st.markdown(f"**{cat.replace('_', ' ').title()}**")
            st.image(cv2.cvtColor(items[0]['bgr'], cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"{len(items)} Reference Views  
~{sum(len(it['kps']) for it in items):,} Total Keypoints")
