import cv2
import dlib
import time
import pyautogui
import numpy as np

from imutils import face_utils
from scipy.spatial import distance as dist
class CursorController:
    def __init__(self, smoothing=0.3, move_step=20, hold_time=1.2):
        self.smoothing = smoothing
        self.move_step = move_step
        self.hold_time = hold_time

        self.prev_x = None
        self.prev_y = None
        self.center_hold_start = None

    def move(self, gaze):
        if gaze in ("UNKNOWN", "CENTER"):
            return

        x, y = pyautogui.position()

        if gaze == "LEFT":
            x -= self.move_step
        elif gaze == "RIGHT":
            x += self.move_step
        elif gaze == "UP":
            y -= self.move_step
        elif gaze == "DOWN":
            y += self.move_step

        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        sx = int(self.prev_x + self.smoothing * (x - self.prev_x))
        sy = int(self.prev_y + self.smoothing * (y - self.prev_y))

        pyautogui.moveTo(sx, sy)
        self.prev_x, self.prev_y = sx, sy

    def gaze_click(self, gaze):
        now = time.time()

        if gaze == "CENTER":
            if self.center_hold_start is None:
                self.center_hold_start = now
            elif now - self.center_hold_start >= self.hold_time:
                pyautogui.click()
                self.center_hold_start = None
        else:
            self.center_hold_start = None

