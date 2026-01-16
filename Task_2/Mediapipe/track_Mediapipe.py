import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pyautogui
import time


EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# MediaPipe setup 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Screen size
screen_w, screen_h = pyautogui.size()

# Blink counters
COUNTER = 0
TOTAL_BLINKS = 0
BLINK_TIMES = []

# Gaze hold 
hold_start_time = None
hold_target = None
HOLD_DURATION = 3  # seconds to select

# Cursor smoothing 
prev_x, prev_y = None, None
SMOOTHING = 0.3  # higher = more smoothing

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_iris_center(landmarks, iris_idx, w, h):
    points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_idx])
    return points.mean(axis=0).astype(int)

def determine_gaze(pupil, eye_box):
    ex, ey, ew, eh = eye_box
    dx = (pupil[0] - (ex + ew/2)) / ew
    dy = (pupil[1] - (ey + eh/2)) / eh
    threshold = 0.15

    if dx < -threshold:
        return "LEFT"
    elif dx > threshold:
        return "RIGHT"
    elif dy < -2*threshold:
        return "UP"
    elif dy > threshold:
        return "DOWN"
    else:
        return "CENTER"

# Define a target zone for selection 
target_x, target_y, target_w, target_h = screen_w//2-50, screen_h//2-50, 25, 25

mobile_cam = "https://192.168.1.42:8080/video"
# cap = cv2.VideoCapture(mobile_cam)
cap = cv2.VideoCapture(0)

cam_w, cam_h = int(cap.get(3)), int(cap.get(4))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame , 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Eyes for blink
        leftEye = np.array([(int(landmarks[i].x*cam_w), int(landmarks[i].y*cam_h)) for i in LEFT_EYE])
        rightEye = np.array([(int(landmarks[i].x*cam_w), int(landmarks[i].y*cam_h)) for i in RIGHT_EYE])

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR)/2.0

        # Blink counting 
        COUNTER, TOTAL_BLINKS
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL_BLINKS += 1
                BLINK_TIMES.append(time.time())
            COUNTER = 0

        # Detect double/triple blink
        now = time.time()
        recent_blinks = [t for t in BLINK_TIMES if now-t < 1.5]
        if len(recent_blinks) >= 3:
            pyautogui.click(button='right')
            BLINK_TIMES.clear()
            print("Right Click")
        elif len(recent_blinks) >= 2:
            pyautogui.click(button='left')
            BLINK_TIMES.clear()
            print("Left Click")

        # Pupil centers
        left_center = get_iris_center(landmarks, LEFT_IRIS, cam_w, cam_h)
        right_center = get_iris_center(landmarks, RIGHT_IRIS, cam_w, cam_h)
        pupil_center = ((left_center + right_center)/2).astype(int)

        # Left Eye bounding box
        lex, ley, lew, leh = cv2.boundingRect(leftEye)
        gazeL = determine_gaze(left_center, (lex, ley, lew, leh))

        # right Eye bounding box
        rex, rey, rew, reh = cv2.boundingRect(rightEye)
        gazeR = determine_gaze(right_center, (rex, rey, rew, reh))

        if gazeL == gazeR:
            print(gazeL)
        
        gaze = gazeL
        # Cursor movement with smoothing
        target_x_pos, target_y_pos = pyautogui.position()
        if gaze == "LEFT":
            target_x_pos -= 20
        elif gaze == "RIGHT":
            target_x_pos += 20
        elif gaze == "UP":
            target_y_pos -= 20
        elif gaze == "DOWN":
            target_y_pos += 20
        # CENTER → hold cursor

        # Smooth movement
        if prev_x is None:
            prev_x, prev_y = target_x_pos, target_y_pos
        smooth_x = int(prev_x + SMOOTHING*(target_x_pos - prev_x))
        smooth_y = int(prev_y + SMOOTHING*(target_y_pos - prev_y))
        pyautogui.moveTo(smooth_x, smooth_y)
        prev_x, prev_y = smooth_x, smooth_y

        # Gaze Hold selection
        if (target_x <= smooth_x <= target_x+target_w) and (target_y <= smooth_y <= target_y+target_h):
            if hold_start_time is None:
                hold_start_time = time.time()
            elif time.time() - hold_start_time >= HOLD_DURATION:
                pyautogui.click()  # Select
                hold_start_time = None  # Reset after click
        else:
            hold_start_time = None

        # Draw pupil & eye contours 
        cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)
        cv2.circle(frame, tuple(pupil_center), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"Gaze: {gaze}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Draw target rectangle
        cv2.rectangle(frame, (target_x, target_y), (target_x+target_w, target_y+target_h), (255,0,0), 2)
        if hold_start_time:
            cv2.putText(frame, f"Holding: {int(time.time() - hold_start_time)}s", (target_x, target_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Eye Control with Gaze-Hold", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
