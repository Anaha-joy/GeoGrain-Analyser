"""
GeoGrain Analyzer ML Rock Classifier v2
Improved Feature Stability
Robust Small Grain Handling
CPU Compatible
"""

import numpy as np
import cv2
import joblib
import os


# =====================================================
# FEATURE EXTRACTION FROM SAM MASK
# =====================================================

def extract_features(mask, image):

    try:

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

        if area < 30:  # ignore extremely tiny noise
            return None

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            return None

        # -----------------------------
        # GEOMETRIC FEATURES
        # -----------------------------

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        solidity = area / hull_area if hull_area > 0 else 0

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0

        # -----------------------------
        # TEXTURE FEATURES (Improved)
        # -----------------------------

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Smooth slightly to reduce noise bias
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = cv2.Laplacian(gray_blur, cv2.CV_64F)

        if np.sum(binary) == 0:
            return None

        texture_val = np.mean(np.abs(laplacian[binary == 1]))

        # -----------------------------
        # COLOR / BRIGHTNESS
        # -----------------------------

        mean_color = cv2.mean(image, mask=binary)

        brightness = (mean_color[0] + mean_color[1] + mean_color[2]) / 3

        # -----------------------------
        # FINAL FEATURE VECTOR
        # -----------------------------

        features = [
            area,
            perimeter,
            circularity,
            solidity,
            aspect_ratio,
            texture_val,
            brightness
        ]

        return features

    except Exception as e:
        return None


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

def load_classifier():

    model_path = "modules/rock_model.pkl"

    if not os.path.exists(model_path):

        print("ML model not found — using geological filter only")
        return None

    try:

        model = joblib.load(model_path)
        print("ML rock classifier loaded")
        return model

    except:
        print("Failed to load ML model")
        return None


classifier = load_classifier()


# =====================================================
# ROCK PREDICTION FUNCTION
# =====================================================

def is_rock(mask, image):

    # If no model → allow geological filter only
    if classifier is None:
        return True

    features = extract_features(mask, image)

    if features is None:
        return False

    features = np.array(features).reshape(1, -1)

    try:
        # Use probability instead of hard classification
        if hasattr(classifier, "predict_proba"):

            prob = classifier.predict_proba(features)[0][1]

            # Balanced threshold
            if prob >= 0.45:
                return True
            else:
                return False

        else:
            prediction = classifier.predict(features)
            return prediction[0] == 1

    except:
        return True
