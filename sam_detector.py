"""
GeoGrain Analyzer Professional v9.0
Synchronized ML-Based Detection
SAM (Sensitive) + ML (Primary Filter)
No conflicting geological shape filtering
"""

import cv2
import numpy as np
import os
import csv

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from modules.report_generator import save_pdf


# =====================================================
# 🔥 VECTOR DRAW FUNCTION
# =====================================================
def draw_vector_boundary(overlay, binary_mask):
    mask_uint8 = (binary_mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    smooth_contours = []
    for cnt in contours:
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        smooth_contours.append(approx)

    cv2.drawContours(overlay, smooth_contours, -1, (0, 255, 0), 2)


# =====================================================
# SAVE CSV
# =====================================================
def save_full_csv(grain_data):

    os.makedirs("output", exist_ok=True)
    path = os.path.abspath("output/grain_full_data.csv")

    with open(path, mode="w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Grain_ID",
            "Area_px",
            "Area_mm2",
            "Perimeter_px",
            "Perimeter_mm",
            "Centroid_X_px",
            "Centroid_Y_px",
            "Centroid_X_mm",
            "Centroid_Y_mm",
            "Equivalent_Diameter_mm",
            "a_axis_px",
            "b_axis_px",
            "a_axis_mm",
            "b_axis_mm",
            "Aspect_Ratio_a_by_b",
            "Orientation_angle_deg",
            "Volume_approx_mm3"
        ])

        writer.writerows(grain_data)

    return path


# =====================================================
# LOAD ML CLASSIFIER
# =====================================================
try:
    from modules.rock_classifier import is_rock
    ML_AVAILABLE = True
    print("ML classifier loaded")
except:
    ML_AVAILABLE = False
    print("ML classifier not available")


# =====================================================
# LOAD SAM MODEL
# =====================================================
def load_sam():

    sam_checkpoint = "sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cpu")

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,   # 🔥 REDUCED (48 → 24 = 4x faster)
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        min_mask_region_area=80
    )

    print("SAM loaded (FAST MODE)")
    return mask_generator


mask_generator = load_sam()


# =====================================================
# 🔥 OVERLAP REMOVAL
# =====================================================
def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


# =====================================================
# MAIN DETECTOR
# =====================================================
def detect_grains(image_path, pixel_to_mm):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Cannot load image")

    # =====================================================
    # 🔥 SMART RESIZE (ONLY FOR SPEED - CONTROLLED)
    # =====================================================
    max_size = 1800   # balanced (fast + accurate)

    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)

        # IMPORTANT: adjust scale
        pixel_to_mm = pixel_to_mm / scale

    # =====================================================

    overlay = image.copy()

    print("Running SAM...")
    masks = mask_generator.generate(image)
    print("Total masks:", len(masks))

    # =====================================================
    # REMOVE OVERLAPPING / DUPLICATE MASKS
    # =====================================================
    filtered_masks = []
    for m in sorted(masks, key=lambda x: x["area"], reverse=True):
        keep = True
        for fm in filtered_masks:
            if mask_iou(m["segmentation"], fm["segmentation"]) > 0.5:
                keep = False
                break
        if keep:
            filtered_masks.append(m)

    masks = filtered_masks
    # =====================================================

    grain_sizes = []
    grain_data = []
    grain_volumes = []
    grain_id = 1

    for mask in masks:

        binary = mask["segmentation"].astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        cnt = contours[0]

        if len(cnt) < 5:
            continue

        area = cv2.contourArea(cnt)

        if area < 30:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # ================= ML FILTER =================
        if ML_AVAILABLE:
            try:
                if not is_rock(mask, image):
                    continue
            except:
                continue

        # ================= MEASUREMENTS =================
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        diameter_px = np.sqrt(4 * area / np.pi)
        diameter_mm = diameter_px * pixel_to_mm

        if diameter_mm <= 0:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (_, _), (axis1, axis2), angle = ellipse

        a_px = max(axis1, axis2)
        b_px = min(axis1, axis2)

        a_mm = a_px * pixel_to_mm
        b_mm = b_px * pixel_to_mm

        if a_mm <= 0 or b_mm <= 0:
            continue

        aspect_ratio = a_px / b_px if b_px > 0 else 0

        volume = a_mm * (b_mm ** 2)

        if volume <= 0:
            continue

        volume = min(volume, 1e10)

        area_mm2 = area * (pixel_to_mm ** 2)
        perimeter_mm = perimeter * pixel_to_mm
        cx_mm = cx * pixel_to_mm
        cy_mm = cy * pixel_to_mm

        grain_sizes.append(diameter_mm)
        grain_volumes.append(volume)

        grain_data.append([
            grain_id,
            area,
            area_mm2,
            perimeter,
            perimeter_mm,
            cx,
            cy,
            cx_mm,
            cy_mm,
            diameter_mm,
            a_px,
            b_px,
            a_mm,
            b_mm,
            aspect_ratio,
            angle,
            volume
        ])

        draw_vector_boundary(overlay, binary)
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

        grain_id += 1

    print("Detected rocks:", len(grain_sizes))

    grain_sizes = np.array(grain_sizes)
    grain_volumes = np.array(grain_volumes)

    if len(grain_sizes) == 0:
        stats = dict(count=0, mean=0, d10=0, d50=0, d84=0, d90=0)
    else:
        stats = dict(
            count=len(grain_sizes),
            mean=float(np.mean(grain_sizes)),
            d10=float(np.percentile(grain_sizes, 10)),
            d50=float(np.percentile(grain_sizes, 50)),
            d84=float(np.percentile(grain_sizes, 84)),
            d90=float(np.percentile(grain_sizes, 90))
        )

    os.makedirs("output", exist_ok=True)

    overlay_path = os.path.abspath("output/grain_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    csv_path = save_full_csv(grain_data)

    pdf_path = save_pdf(
        grain_sizes,
        stats,
        overlay_path,
        csv_path,
        grain_volumes
    )

    return dict(
        overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        overlay_path=overlay_path,
        grain_sizes=grain_sizes.tolist(),
        grain_volumes=grain_volumes.tolist(),
        stats=stats,
        csv_path=csv_path,
        pdf_path=pdf_path
    )