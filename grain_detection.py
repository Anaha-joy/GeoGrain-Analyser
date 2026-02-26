import cv2
import numpy as np
import os
import csv

from modules.report_generator import save_pdf


# -------------------------------------------------------
# SAVE FULL CSV
# -------------------------------------------------------
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
            "Volume_approx_mm3"   # 🔥 NEW COLUMN
        ])

        writer.writerows(grain_data)

    print("Full CSV saved:", path)

    return path


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def detect_grains(image_path, pixel_to_mm):

    print("Loading image...")

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Cannot load image")

    max_size = 1600
    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        pixel_to_mm = pixel_to_mm / scale

    overlay = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    vegetation_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_veg_mask = cv2.bitwise_not(vegetation_mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        3
    )

    thresh = cv2.bitwise_and(thresh, thresh, mask=non_veg_mask)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    grain_sizes = []
    grain_data = []
    grain_volumes = []   # 🔥 NEW
    grain_id = 1

    print("Detecting rocks...")

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 80 or area > 100000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.25:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        diameter_px = np.sqrt(4 * area / np.pi)
        diameter_mm = diameter_px * pixel_to_mm

        if len(cnt) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (x_center, y_center), (axis1, axis2), angle = ellipse

        if axis1 <= 0 or axis2 <= 0:
            continue

        a_px = max(axis1, axis2)
        b_px = min(axis1, axis2)

        a_mm = a_px * pixel_to_mm
        b_mm = b_px * pixel_to_mm

        aspect_ratio = a_px / b_px if b_px > 0 else 0

        # 🔥 VOLUME APPROXIMATION (BASEGRAIN STYLE)
        volume = a_mm * (b_mm ** 2)

        # Unit conversions
        area_mm2 = area * (pixel_to_mm ** 2)
        perimeter_mm = perimeter * pixel_to_mm
        cx_mm = cx * pixel_to_mm
        cy_mm = cy * pixel_to_mm

        grain_sizes.append(diameter_mm)
        grain_volumes.append(volume)   # 🔥 NEW

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
            volume   # 🔥 SAVED
        ])

        cv2.drawContours(overlay, [cnt], -1, (0,255,0), 2)
        cv2.circle(overlay, (cx,cy), 3, (0,0,255), -1)

        grain_id += 1

    print("Detected rocks:", len(grain_sizes))

    grain_sizes = np.array(grain_sizes)
    grain_volumes = np.array(grain_volumes)  # 🔥 NEW

    if len(grain_sizes) == 0:
        stats = dict(count=0, mean=0, d10=0, d50=0, d84=0, d90=0)
    else:
        stats = dict(
            count=len(grain_sizes),
            mean=float(np.mean(grain_sizes)),
            d10=float(np.percentile(grain_sizes,10)),
            d50=float(np.percentile(grain_sizes,50)),
            d84=float(np.percentile(grain_sizes,84)),
            d90=float(np.percentile(grain_sizes,90))
        )

    os.makedirs("output", exist_ok=True)

    overlay_path = os.path.abspath("output/grain_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    csv_path = save_full_csv(grain_data)

    # 🔥 PASS VOLUME TO REPORT
    pdf_path = save_pdf(grain_sizes, stats, overlay_path, csv_path, grain_volumes)

    return dict(
        overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        overlay_path=overlay_path,
        grain_sizes=grain_sizes.tolist(),
        grain_volumes=grain_volumes.tolist(),  # 🔥 NEW
        stats=stats,
        csv_path=csv_path,
        pdf_path=pdf_path
    )