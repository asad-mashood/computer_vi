"""
Computer Vision Lab Assignment 06: Landmark Recognition Using SIFT Algorithm
Student Name: Muhammad Asad Mashood
Student ID: FA23-BAI-039
Repository: ComputerVision-LABS / lab06 / Landmark_Recognition_SIFT
Platform: Python 3.10+ | OpenCV 4.8+ / 5.0+ | NumPy | Matplotlib
"""

import os
import glob
import time
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Task 1: Landmark Database Ingestion
def task1_load_database(data_dir="data/landmarks"):
    print("[Task 1] Loading Landmark Database...")
    db = {}
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != 'unknown_landmark']
    for cat in sorted(categories):
        p = os.path.join(data_dir, cat)
        files = sorted(glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.png")))
        db[cat] = []
        for f in files:
            bgr = cv2.imread(f)
            if bgr is not None:
                db[cat].append({
                    'file': os.path.basename(f),
                    'path': f,
                    'bgr': bgr
                })
    return db

# Task 2: SIFT Feature Extraction
def task2_extract_features(db):
    print("[Task 2] Extracting SIFT Keypoints and 128-D Descriptors...")
    sift = cv2.SIFT_create()
    for cat, items in db.items():
        for item in items:
            gray = cv2.cvtColor(item['bgr'], cv2.COLOR_BGR2GRAY)
            kps, descs = sift.detectAndCompute(gray, None)
            item['gray'] = gray
            item['keypoints'] = kps
            item['descriptors'] = descs
    return db

# Task 3: Query Processing
def task3_process_query(query_bgr, sift):
    gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY) if len(query_bgr.shape) == 3 else query_bgr
    kps, descs = sift.detectAndCompute(gray, None)
    return gray, kps, descs

# Task 4: Feature Matching & Landmark Recognition
def task4_match_and_rank(des_q, db, ratio_threshold=0.75, min_match_threshold=20):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ranked = []

    for cat, items in db.items():
        best_matches = []
        best_ref = None
        for item in items:
            if des_q is None or item['descriptors'] is None or len(des_q) < 2 or len(item['descriptors']) < 2:
                continue
            knn = bf.knnMatch(des_q, item['descriptors'], k=2)
            good = [m for m, n in knn if m.distance < ratio_threshold * n.distance]
            if len(good) > len(best_matches):
                best_matches = good
                best_ref = item

        ranked.append({
            'landmark': cat,
            'match_count': len(best_matches),
            'matches': best_matches,
            'best_ref': best_ref
        })

    ranked.sort(key=lambda x: x['match_count'], reverse=True)
    top = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None

    is_rec = False
    status_msg = ""
    if top is None or top['match_count'] < min_match_threshold:
        is_rec = False
        status_msg = f"Rejected: Best match count ({top['match_count'] if top else 0}) below threshold {min_match_threshold}"
    elif second and second['match_count'] > 0 and (top['match_count'] / second['match_count'] < 1.35) and top['match_count'] < 35:
        is_rec = False
        status_msg = f"Rejected: Ambiguous match between {top['landmark']} and {second['landmark']}"
    else:
        is_rec = True
        status_msg = f"Recognized: {top['landmark']} ({top['match_count']} reliable matches)"

    return is_rec, top, ranked, status_msg

# Task 5: Landmark Localization
def task5_localize(query_bgr, kps_q, ref_bgr, kps_ref, matches, min_inliers=12):
    h_q, w_q = query_bgr.shape[:2]
    h_r, w_r = ref_bgr.shape[:2]
    ann = query_bgr.copy()

    if len(matches) < 4:
        return False, 0, ann, "Insufficient matches (< 4)"

    src_pts = np.float32([kps_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_q[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return False, 0, ann, "Homography estimation failed"

    inliers = int(mask.sum())
    if inliers < min_inliers or (inliers / len(matches)) < 0.20:
        _banner(ann, f"GEOMETRIC VERIFICATION FAILED (Inliers {inliers} < {min_inliers})", (0, 0, 255))
        return False, inliers, ann, f"Inliers {inliers} < {min_inliers}"

    det = np.linalg.det(H[:2, :2])
    if det <= 0:
        _banner(ann, "GEOMETRIC VERIFICATION FAILED (Negative Determinant)", (0, 0, 255))
        return False, inliers, ann, "Reflective/Mirrored mapping"

    corners = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)

    if not cv2.isContourConvex(np.int32(proj)):
        _banner(ann, "GEOMETRIC VERIFICATION FAILED (Non-Convex Polygon)", (0, 0, 255))
        return False, inliers, ann, "Non-convex projected polygon"

    area = cv2.contourArea(np.int32(proj))
    q_area = w_q * h_q
    if area < 0.02 * q_area or area > 3.0 * q_area:
        _banner(ann, "GEOMETRIC VERIFICATION FAILED (Invalid Area Ratio)", (0, 0, 255))
        return False, inliers, ann, "Degenerate polygon area"

    # Draw localized polygon
    pts = np.int32(proj)
    cv2.polylines(ann, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(ann, tuple(p[0]), 5, (0, 200, 255), -1)

    _banner(ann, f"LOCALIZED: {inliers}/{len(matches)} inliers", (0, 255, 0))
    return True, inliers, ann, "Localization success"

def _banner(img, text, color):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (0, 0), (w, 42), color, 2)
    cv2.putText(img, text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2)

# Task 6: Performance Evaluation
def main():
    print("=" * 85)
    print("  COMPUTER VISION LAB 06: SIFT LANDMARK RECOGNITION (MUHAMMAD ASAD MASHOOD - FA23-BAI-039)")
    print("=" * 85)

    sift = cv2.SIFT_create()
    db = task1_load_database()
    db = task2_extract_features(db)

    # Plot Task 1 Database
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor('#1e293b')
    for ax, (cat, items) in zip(axes, sorted(db.items())):
        ax.set_facecolor('#0f172a')
        rgb = cv2.cvtColor(items[0]['bgr'], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{cat.replace('_', ' ').title()}\n({len(items)} Views)", color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/task1_reference_database.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # Plot Task 2 Keypoints
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor('#1e293b')
    for ax, (cat, items) in zip(axes, sorted(db.items())):
        ax.set_facecolor('#0f172a')
        item = items[0]
        vis = cv2.drawKeypoints(item['bgr'], item['keypoints'][:400], None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{cat.replace('_', ' ').title()}\n{len(item['keypoints'])} SIFT KPs", color='#4ade80', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/task2_sift_features.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # Task 6 Benchmarks
    eiffel_ref = db['eiffel_tower'][0]['bgr']
    eiffel_view3 = db['eiffel_tower'][2]['bgr']
    big_ben = cv2.imread("data/landmarks/unknown_landmark/big_ben_1.jpg")

    h, w = eiffel_ref.shape[:2]
    q_orig = eiffel_ref.copy()
    M = cv2.getRotationMatrix2D((w/2, h/2), 45, 0.85)
    q_rot = cv2.warpAffine(eiffel_ref, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    q_scale = cv2.resize(eiffel_ref, (w//2, h//2), interpolation=cv2.INTER_AREA)
    q_illum = np.clip(eiffel_ref.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    q_view = eiffel_view3.copy()
    q_unknown = big_ben.copy()

    test_cases = [
        ("Original Landmark Image", q_orig, "eiffel_tower"),
        ("Rotated Landmark Image (45 deg)", q_rot, "eiffel_tower"),
        ("Resized Landmark Image (0.5x Scale)", q_scale, "eiffel_tower"),
        ("Reduced Brightness Image (0.35x)", q_illum, "eiffel_tower"),
        ("Different Viewpoint Image", q_view, "eiffel_tower"),
        ("Non-Database Landmark (Big Ben)", q_unknown, "unknown")
    ]

    records = []
    matrix_imgs = []
    matches_vis = []

    print("\n" + "-" * 110)
    print(f"{'Condition':<28} | {'Ground Truth':<14} | {'Predicted':<18} | {'Matches':<8} | {'Inliers':<8} | {'Decision':<10} | {'Time (ms)':<9}")
    print("-" * 110)

    for name, q_img, gt in test_cases:
        t0 = time.time()
        gray_q, kps_q, des_q = task3_process_query(q_img, sift)
        is_rec, top, ranked, msg = task4_match_and_rank(des_q, db)

        if is_rec and top['best_ref']:
            success, inliers, ann_img, loc_msg = task5_localize(
                q_img, kps_q,
                top['best_ref']['bgr'], top['best_ref']['keypoints'],
                top['matches']
            )
            dec = "RECOGNIZED"
        else:
            inliers = 0
            ann_img = q_img.copy()
            _banner(ann_img, "REJECTED [NON-DATABASE / UNKNOWN]", (0, 0, 255))
            dec = "REJECTED"

        dt_ms = (time.time() - t0) * 1000
        n_m = top['match_count'] if top else 0
        pred_label = top['landmark'] if is_rec else 'Unknown / Rejected'

        records.append({
            'Condition': name,
            'Ground_Truth': gt,
            'Predicted_Landmark': pred_label,
            'Query_Keypoints': len(kps_q),
            'Reliable_Matches': n_m,
            'RANSAC_Inliers': inliers,
            'Decision': dec,
            'Processing_Time_ms': round(dt_ms, 2)
        })

        matrix_imgs.append((name, ann_img, dec, n_m, inliers))

        if name in ["Rotated Landmark Image (45 deg)", "Non-Database Landmark (Big Ben)"] and top and top['best_ref']:
            m_vis = cv2.drawMatches(
                q_img, kps_q,
                top['best_ref']['bgr'], top['best_ref']['keypoints'],
                top['matches'][:50], None,
                matchColor=(0, 255, 0) if is_rec else (0, 0, 255),
                singlePointColor=(180, 180, 180),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            matches_vis.append((name, m_vis, is_rec, n_m))

        print(f"{name:<28} | {gt:<14} | {pred_label:<18} | {n_m:<8} | {inliers:<8} | {dec:<10} | {dt_ms:<9.1f}")

    print("-" * 110)

    # Plot Task 4 Matching
    fig, axes = plt.subplots(len(matches_vis), 1, figsize=(15, 5.5 * len(matches_vis)))
    if len(matches_vis) == 1:
        axes = [axes]
    fig.patch.set_facecolor('#1e293b')
    for ax, (c_name, m_bgr, is_rec, n_m) in zip(axes, matches_vis):
        ax.set_facecolor('#0f172a')
        ax.imshow(cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB))
        col = '#4ade80' if is_rec else '#f87171'
        txt = f"[RECOGNIZED: {n_m} reliable matches]" if is_rec else f"[REJECTED: {n_m} spurious matches]"
        ax.set_title(f"Task 4: Feature Matches for {c_name} -> {txt}", color=col, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/task4_feature_matching_comparison.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # Plot Task 6 Environmental Matrix
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#1e293b')
    for ax, (c_name, ann, dec, n_m, n_in) in zip(axes.flatten(), matrix_imgs):
        ax.set_facecolor('#0f172a')
        ax.imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
        col = '#4ade80' if dec == 'RECOGNIZED' else '#f87171'
        ax.set_title(f"{c_name}\nDecision: {dec} | Matches: {n_m} | Inliers: {n_in}", color=col, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("figures/task6_environmental_evaluation.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # Plot Task 6 Summary
    conds = [r['Condition'] for r in records]
    matches = [r['Reliable_Matches'] for r in records]
    inliers = [r['RANSAC_Inliers'] for r in records]
    times = [r['Processing_Time_ms'] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#1e293b')

    x = np.arange(len(conds))
    width = 0.35
    ax1.set_facecolor('#0f172a')
    ax1.bar(x - width/2, matches, width, label='Reliable Matches', color='#38bdf8')
    ax1.bar(x + width/2, inliers, width, label='RANSAC Inliers', color='#4ade80')
    ax1.set_title('Reliable Matches & RANSAC Inliers', color='#f1f5f9', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds, rotation=25, ha='right', color='#cbd5e1', fontsize=9)
    ax1.tick_params(colors='#cbd5e1')
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9')
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#475569')

    ax2.set_facecolor('#0f172a')
    bars = ax2.bar(x, times, width*1.5, color='#c084fc')
    ax2.set_title('Processing Latency (ms)', color='#f1f5f9', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conds, rotation=25, ha='right', color='#cbd5e1', fontsize=9)
    ax2.tick_params(colors='#cbd5e1')
    ax2.grid(axis='y', linestyle='--', alpha=0.3, color='#475569')
    for b in bars:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 1.5, f'{yval:.1f}ms', ha='center', va='bottom', color='#e2e8f0', fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/task6_benchmark_telemetry.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # Save JSON & CSV
    with open("results/lab06_experimental_data.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    with open("results/lab06_experimental_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print("\n[Success] Muhammad Asad Mashood Lab 06 execution completed successfully!")

if __name__ == '__main__':
    main()
