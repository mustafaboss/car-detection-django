import cv2 as cv
import numpy as np

# Minimum width and height of detected rectangles
min_width_rect = 80
min_height_rect = 80
count_line_position = 550

# Web camera or video file
cap = cv.VideoCapture("video.mp4")

# Initialize background subtractor
algo = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6  # Allowable error between pixels
counter = 0

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    # Convert to grayscale and blur the frame
    grey = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5, 5), 5)

    # Apply background subtractor
    img_sub = algo.apply(blur)

    # Apply morphological operations to reduce noise
    dilat = cv.dilate(img_sub, np.ones((5, 5), np.uint8))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilatada = cv.morphologyEx(dilat, cv.MORPH_CLOSE, kernel)
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for contour in contours:
        # Filter out small contours
        (x, y, w, h) = cv.boundingRect(contour)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue

        # Draw rectangles around detected objects
        cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv.circle(frame1, center, 4, (0, 0, 255), -1)

    for (x, y) in detect:
        if count_line_position - offset < y < count_line_position + offset:
            counter += 1
            cv.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            detect.remove((x, y))
            print("Vehicle counter:", counter)

    # Display the vehicle count
    cv.putText(frame1, "Vehicle Counter: " + str(counter), (450, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Display the video with detected objects
    cv.imshow('Vehicle Detection', frame1)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
