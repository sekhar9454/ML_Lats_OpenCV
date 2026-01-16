import cv2 as cv
import numpy as np



# setting vlaues of constants -> hyperParameters
URL = "https://192.0.0.4:8080/video"
MIN_CONTOUR_AREA = 1000 # helpd in ignoring small objects 
TRAJECTORY_LENGTH = 50

GAUSSIAN_KERNEL = (5, 5)
CANNY_THRESHOLDS = (50, 150)
MORPH_KERNEL_SIZE = (5, 5)



# Frame Processing -> Gray conversion + Gaussian Blur
def preprocess_frame(frame):
    """
    Convert frame to grayscale and apply Gaussian blur.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    return blurred


def detect_edges(blurred_frame):
    """
    Perform Canny edge detection and morphological closing.
    """
    edges = cv.Canny(blurred_frame, *CANNY_THRESHOLDS)
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    return edges, edges_closed



# Contour Processing and filtering small areas which are having area less than 1000
def get_largest_contour(edges_closed):
    """
    Find and return the largest contour above area threshold.
    """
    contours, _ = cv.findContours(
        edges_closed,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv.contourArea)
    if cv.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    return largest


def get_centroid(contour):
    """
    Compute centroid of a contour.
    """
    M = cv.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


# Drawing over original frame
def draw_object_features(frame, contour):
    """
    Draw contour, bounding box, and convex hull.
    """
    # Contour
    cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Bounding box
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convex hull
    hull = cv.convexHull(contour)
    cv.drawContours(frame, [hull], -1, (0, 0, 255), 2)


def draw_trajectory(frame, trajectory):
    """
    Draw motion trajectory.
    """
    for i in range(1, len(trajectory)):
        cv.line(frame, trajectory[i - 1], trajectory[i], (255, 255, 0), 2)



def getBoundryTracker():
    webcam = cv.VideoCapture(0)
    trajectory = []

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to get frame")
            break

        blurred = preprocess_frame(frame)
        edges, edges_closed = detect_edges(blurred)


        # Contour detection
        contour = get_largest_contour(edges_closed)
        if contour is not None:
            draw_object_features(frame, contour)

            centroid = get_centroid(contour)
            if centroid:
                trajectory.append(centroid)
                trajectory = trajectory[-TRAJECTORY_LENGTH:]

        # Drawing trajectory over original frame
        draw_trajectory(frame, trajectory)

        cv.imshow("Object Boundary Tracker", frame)

        if cv.waitKey(1) & 0xFF == 27:  # 27 corresponds to ESC
            break

    webcam.release()
    cv.destroyAllWindows()


getBoundryTracker()
