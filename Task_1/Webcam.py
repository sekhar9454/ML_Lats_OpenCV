import cv2 as cv
import numpy as np

URL = 'https://192.0.0.4:8080/video'

webcam = cv.VideoCapture(0) # Webcam
# webcam = cv.VideoCapture(URL) # Mobile device cam -> can use via 'IP WEBCAM'    

trajectory = []

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # 1. Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    gaussian_blur = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow('Gaussian Blur frame' , gaussian_blur)

    # 3. Canny Edge Detection
    edges = cv.Canny(gaussian_blur, 50, 150)
    cv.imshow('Initial Edges' , edges)

    # 4. Morphological Closing (Edge Stabilization)
    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    cv.imshow('Closed Edges' , edges_closed)

    # # Thickens the edges
    # dialated_frame = cv.dilate(edges , (5,5) , iterations=3)
    # cv.imshow('Dialated Frame' ,dialated_frame )

    # # Removes the noise
    # eroded_frame = cv.erode(dialated_frame ,(5,5) , None , iterations=3 )
    # cv.imshow('Eroded Frame' , eroded_frame)

    # edges_closed = eroded_frame

    # 5. Find Contours
    contours, _ = cv.findContours(
        edges_closed,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # 6. Select largest contour
        cnt = max(contours, key=cv.contourArea)

        # Ignore small noise
        if cv.contourArea(cnt) > 1000:

            # Draw contour
            cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            # Bounding box
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Convex hull
            hull = cv.convexHull(cnt)
            cv.drawContours(frame, [hull], -1, (0, 0, 255), 2)

            # Centroid
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                trajectory.append((cx, cy))
                if len(trajectory) > 50:
                    trajectory.pop(0)

    # Draw trajectory
    for i in range(1, len(trajectory)):
        cv.line(frame, trajectory[i - 1], trajectory[i], (255, 255, 0), 2)

    cv.imshow("Object Boundary Tracker", frame)

    if cv.waitKey(1) & 0xFF == 27:  # 27 corresponds to ESC
        break

webcam.release()
cv.destroyAllWindows()
