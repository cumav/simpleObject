import time

import cv2

from SimpleObjectDetection import SimpleObjectDetection


def get_center(small_value, big_value):
    return int(((big_value - small_value)/2) + small_value)

a = SimpleObjectDetection()

points = []

# Open the video stream. With a number you select a webcam. With a filepath you select a video file
cap = cv2.VideoCapture("tet.mp4")
while cap.isOpened():
    # Retrieve frames
    ret, frame = cap.read()
    # Detect the features
    features = a.return_detection_features(frame)
    for detection in features:
        # Get all items that have a confidence of 10% or more
        if features[detection]["score"] >= 0.3 and features[detection]["name"] == "person":
            width_center = get_center(features[detection]["left"],
                                      features[detection]["right"])
            height_center = get_center(features[detection]["top"],
                                       features[detection]["bottom"])
            points.append((width_center, height_center))

    for item in points:
        cv2.circle(frame, item, 5, (0, 255, 0), -1)

    if True:
        cv2.imshow('img', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            pass
