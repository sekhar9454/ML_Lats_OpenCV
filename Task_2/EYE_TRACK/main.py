# Complete OOps based for clear understading
import cv2
import dlib
import time
import pyautogui
import numpy as np

from imutils import face_utils
from scipy.spatial import distance as dist


from blink_detector import BlinkDetector
from cursor_controller import CursorController
from gaze_tracker import GazeTracker



class EyeTrackerApp:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "../Bonus/Model/shape_predictor_68_face_landmarks.dat"
        )

        self.gaze_tracker = GazeTracker()
        self.blink_detector = BlinkDetector()
        self.cursor = CursorController()

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.cap = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

            faces = self.detector(gray, 0)

            if faces:
                face = faces[0]
                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]

                ear = (
                    self.gaze_tracker.eye_aspect_ratio(leftEye) +
                    self.gaze_tracker.eye_aspect_ratio(rightEye)
                ) / 2.0

                action = self.blink_detector.update(ear)

                if action == "DOUBLE":
                    pyautogui.click(button="left")
                elif action == "TRIPLE":
                    pyautogui.click(button="right")

                (x, y, w, h) = cv2.boundingRect(leftEye)
                leftImg = frame[y:y+h, x:x+w]

                (x2, y2, w2, h2) = cv2.boundingRect(rightEye)
                rightImg = frame[y2:y2+h2, x2:x2+w2]

                leftPupil = self.gaze_tracker.detect_pupil(leftImg)
                rightPupil = self.gaze_tracker.detect_pupil(rightImg)

                gaze = "UNKNOWN"
                if leftPupil and rightPupil:
                    g1 = self.gaze_tracker.gaze_direction(leftPupil, w, h)
                    g2 = self.gaze_tracker.gaze_direction(rightPupil, w2, h2)
                    if g1 == g2:
                        gaze = g1
                        self.cursor.move(gaze)
                        self.cursor.gaze_click(gaze)

                cv2.putText(frame, f"Gaze: {gaze}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                cv2.putText(frame,
                            f"Blinks: {self.blink_detector.total_blinks}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)

            cv2.imshow("Eye Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    EyeTrackerApp().run()

