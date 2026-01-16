import cv2 as cv
import dlib
import numpy as np
import time
import pyautogui

from imutils import face_utils
from scipy.spatial import distance as dist


# EAR FUNCTION 
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# PUPIL DETECTION
def detectPupil(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None

    gray = cv.cvtColor(eye_img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    pupil = max(contours, key=cv.contourArea)
    M = cv.moments(pupil)

    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)


# GAZE DIRECTION 
def detectGaze(pupil, w, h):
    if pupil is None or w == 0 or h == 0:
        return "UNKNOWN"

    x, y = pupil
    xr = x / w
    yr = y / h

    if xr < 0.35:
        return "LEFT"
    elif xr > 0.65:
        return "RIGHT"
    elif yr < 0.35:
        return "UP"
    elif yr > 0.65:
        return "DOWN"
    else:
        return "CENTER"


# CURSOR CONTROL
prev_x, prev_y = None, None
SMOOTHING = 0.3
MOVE_STEP = 20

def directCursor(gaze):
    global prev_x, prev_y

    if gaze == "UNKNOWN" or gaze == "CENTER":
        return

    x, y = pyautogui.position()

    if gaze == "LEFT":
        x -= MOVE_STEP
    elif gaze == "RIGHT":
        x += MOVE_STEP
    elif gaze == "UP":
        y -= MOVE_STEP
    elif gaze == "DOWN":
        y += MOVE_STEP

    if prev_x is None:
        prev_x, prev_y = x, y

    smooth_x = int(prev_x + SMOOTHING * (x - prev_x))
    smooth_y = int(prev_y + SMOOTHING * (y - prev_y))

    pyautogui.moveTo(smooth_x, smooth_y)

    prev_x, prev_y = smooth_x, smooth_y


# BLINK PROCESS FUNCTION
def process_blink(ear):
    global COUNTER, TOTAL_BLINKS, BLINK_TIMES, last_click_time

    now = time.time()

    if ear < EYE_AR_THRESH:
        COUNTER += 1
    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            TOTAL_BLINKS += 1
            BLINK_TIMES.append(now)
        COUNTER = 0

    BLINK_TIMES = [t for t in BLINK_TIMES if now - t < 1.5]

    if now - last_click_time > CLICK_COOLDOWN:
        if len(BLINK_TIMES) >= 3:
            pyautogui.click(button="right")
            last_click_time = now
            BLINK_TIMES.clear()
        elif len(BLINK_TIMES) == 2:
            pyautogui.click(button="left")
            last_click_time = now
            BLINK_TIMES.clear()


# GAZE CLICK
gaze_hold_start = None
GAZE_HOLD_TIME = 1.2  # seconds

def gaze_click(gaze):
    global gaze_hold_start

    now = time.time()

    if gaze == "CENTER":
        if gaze_hold_start is None:
            gaze_hold_start = now
        elif now - gaze_hold_start >= GAZE_HOLD_TIME:
            pyautogui.click()
            gaze_hold_start = None
    else:
        gaze_hold_start = None


# CONSTANTS
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL_BLINKS = 0
BLINK_TIMES = []

CLICK_COOLDOWN = 1.0
last_click_time = 0


# DLIB MODELS
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# CAMERA
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame , 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 1.5)

    faces = detector(gray, 0)

    if faces:
        face = faces[0]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        # BLINK HANDLING
        process_blink(ear)

        # EYE ROIs
        (x, y, w, h) = cv.boundingRect(leftEye)
        leftEyeImg = frame[y:y+h, x:x+w]

        (x2, y2, w2, h2) = cv.boundingRect(rightEye)
        rightEyeImg = frame[y2:y2+h2, x2:x2+w2]

        if leftEyeImg.size == 0 or rightEyeImg.size == 0:
            continue

        leftPupil = detectPupil(leftEyeImg)
        rightPupil = detectPupil(rightEyeImg)

        gaze = "UNKNOWN"
        if leftPupil is not None and rightPupil is not None:
            g1 = detectGaze(leftPupil, leftEyeImg.shape[1], leftEyeImg.shape[0])
            g2 = detectGaze(rightPupil, rightEyeImg.shape[1], rightEyeImg.shape[0])
            if g1 == g2:
                gaze = g1
                directCursor(gaze)
                gaze_click(gaze)

        # DRAW 
        cv.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
        cv.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)

        cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv.putText(frame, f"EAR: {ear:.2f}", (20, 80),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv.putText(frame, f"Gaze: {gaze}", (20, 120),
                    cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv.imshow("Left Eye", leftEyeImg)
        cv.imshow("Right Eye", rightEyeImg)

    cv.imshow("Dlib Eye Tracker", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
