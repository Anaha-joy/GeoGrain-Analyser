"""
GeoGrain Analyzer Professional v3 Trainer
Stable Layout + Controlled Display Size
Synchronized With Detector
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import joblib
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =====================================================
# LOAD SAM (Same as Detector)
# =====================================================

def load_sam():

    checkpoint = "sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cpu")

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=48,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        min_mask_region_area=80
    )

    return mask_generator


mask_generator = load_sam()


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(mask, image):

    binary = mask["segmentation"].astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = contours[0]

    area = cv2.contourArea(cnt)
    if area < 30:
        return None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return None

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(gray_blur, cv2.CV_64F)
    texture_val = np.mean(np.abs(lap[binary == 1]))

    mean_color = cv2.mean(image, mask=binary)
    brightness = sum(mean_color[:3]) / 3

    return [
        area,
        perimeter,
        circularity,
        solidity,
        aspect_ratio,
        texture_val,
        brightness
    ]


# =====================================================
# TRAINER GUI
# =====================================================

class RockTrainerGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("GeoGrain Rock Classifier Trainer")
        self.root.geometry("1000x800")

        self.image = None
        self.masks = []
        self.index = 0

        self.features = []
        self.labels = []

        self.build_ui()


    # ---------------- UI ----------------

    def build_ui(self):

        top = tk.Frame(self.root)
        top.pack(pady=10)

        tk.Button(top, text="Load Image", command=self.load_image, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Save Model", command=self.save_model, width=15).pack(side=tk.LEFT, padx=5)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        bottom = tk.Frame(self.root)
        bottom.pack(pady=15)

        tk.Button(bottom, text="ROCK", bg="green", fg="white",
                  width=18, height=2, command=self.label_rock).pack(side=tk.LEFT, padx=20)

        tk.Button(bottom, text="NON-ROCK", bg="red", fg="white",
                  width=18, height=2, command=self.label_nonrock).pack(side=tk.LEFT, padx=20)

        # -------- LIVE COUNTERS --------
        self.counter_label = tk.Label(
            self.root,
            text="ROCK: 0   |   NON-ROCK: 0",
            font=("Arial", 12, "bold")
        )
        self.counter_label.pack(pady=5)

        self.total_label = tk.Label(
            self.root,
            text="Total Labeled: 0",
            font=("Arial", 11)
        )
        self.total_label.pack()

        self.status = tk.Label(self.root,
                               text="Load image to start training",
                               font=("Arial", 12))
        self.status.pack(pady=10)


    # ---------------- LOAD IMAGE ----------------

    def load_image(self):

        path = filedialog.askopenfilename()
        if not path:
            return

        image = cv2.imread(path)
        self.image = image

        masks = mask_generator.generate(image)

        masks = sorted(masks, key=lambda m: m["area"])

        img_area = image.shape[0] * image.shape[1]
        self.masks = [m for m in masks if m["area"] < 0.4 * img_area]

        self.index = 0
        self.show_mask()


    # ---------------- DISPLAY MASK ----------------

    def show_mask(self):

        if self.index >= len(self.masks):
            self.status.config(text="Training complete. Click Save Model.")
            return

        mask = self.masks[self.index]
        binary = mask["segmentation"].astype(np.uint8)

        overlay = self.image.copy()

        dim = (overlay * 0.35).astype(np.uint8)
        overlay[binary == 0] = dim[binary == 0]

        contours, _ = cv2.findContours(binary,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        max_display_width = 1000
        max_display_height = 600

        h, w = overlay.shape[:2]

        scale = min(
            max_display_width / w,
            max_display_height / h,
            1.0
        )

        overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(overlay)
        imgtk = ImageTk.PhotoImage(img)

        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk

        self.status.config(
            text=f"Segment {self.index+1}/{len(self.masks)} | Area: {int(mask['area'])} px"
        )


    # ---------------- LABELING ----------------

    def label_rock(self):
        self.add_label(1)

    def label_nonrock(self):
        self.add_label(0)

    def add_label(self, label):

        mask = self.masks[self.index]
        feat = extract_features(mask, self.image)

        if feat is not None:
            self.features.append(feat)
            self.labels.append(label)

        self.index += 1

        # -------- UPDATE COUNTERS --------
        rock_count = self.labels.count(1)
        nonrock_count = self.labels.count(0)
        total = len(self.labels)

        self.counter_label.config(
            text=f"ROCK: {rock_count}   |   NON-ROCK: {nonrock_count}"
        )

        self.total_label.config(
            text=f"Total Labeled: {total}"
        )

        self.show_mask()


    # ---------------- SAVE MODEL ----------------

    def save_model(self):

        if len(self.features) < 40:
            messagebox.showerror("Error", "Need at least 40 labeled samples")
            return

        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42
        )

        clf.fit(self.features, self.labels)

        os.makedirs("modules", exist_ok=True)
        joblib.dump(clf, "modules/rock_model.pkl")

        messagebox.showinfo("Success", "Model saved successfully")


# =====================================================
# MAIN
# =====================================================

def main():
    root = tk.Tk()
    app = RockTrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()