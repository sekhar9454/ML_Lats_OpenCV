import cv2
import dlib
import time
import pyautogui
import numpy as np

from imutils import face_utils
from scipy.spatial import distance as dist
class BlinkDetector:
    def __init__(
        self,
        ear_thresh=0.25,
        consec_frames=3,
        blink_window=1.5,
        cooldown=1.0
    ):
        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.blink_window = blink_window
        self.cooldown = cooldown

        self.frame_counter = 0
        self.blink_times = []
        self.last_action_time = 0
        self.total_blinks = 0

    def update(self, ear):
        now = time.time()

        # Detect blink
        if ear < self.ear_thresh:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.total_blinks += 1
                self.blink_times.append(now)
            self.frame_counter = 0

        # Remove old blinks
        self.blink_times = [
            t for t in self.blink_times if now - t <= self.blink_window
        ]

        # Cooldown
        if now - self.last_action_time < self.cooldown:
            return None

        if len(self.blink_times) >= 3:
            self.blink_times.clear()
            self.last_action_time = now
            return "TRIPLE"

        if len(self.blink_times) == 2:
            self.blink_times.clear()
            self.last_action_time = now
            return "DOUBLE"

        return None
