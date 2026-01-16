import cv2
import dlib
import time
import pyautogui
import numpy as np

from imutils import face_utils
from scipy.spatial import distance as dist
class GazeTracker:
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return 0 if C == 0 else (A + B) / (2.0 * C)

    def detect_pupil(self, eye_img):
        if eye_img is None or eye_img.size == 0:
            return None

        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        pupil = max(contours, key=cv2.contourArea)
        M = cv2.moments(pupil)

        if M["m00"] == 0:
            return None

        return (
            int(M["m10"] / M["m00"]),
            int(M["m01"] / M["m00"])
        )

    def gaze_direction(self, pupil, w, h):
        if pupil is None or w == 0 or h == 0:
            return "UNKNOWN"

        x, y = pupil
        xr, yr = x / w, y / h

        if xr < 0.35:
            return "LEFT"
        if xr > 0.65:
            return "RIGHT"
        if yr < 0.35:
            return "UP"
        if yr > 0.65:
            return "DOWN"
        return "CENTER"
