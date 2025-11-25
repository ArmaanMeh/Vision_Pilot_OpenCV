import cv2
import numpy as np

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier("Haar_Cascades XML/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("Haar_Cascades XML/haarcascade_eye.xml")

# HSV ranges for red
lower_red1 = np.array([0, 80, 60])
upper_red1 = np.array([10, 240, 230])
lower_red2 = np.array([170, 80, 60])
upper_red2 = np.array([180, 240, 230])

kernelo = np.ones((5, 5))
kernelc = np.ones((10, 10))

def detect_objects(img):
    results = []  # [(tag, cx, cy), ...]

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Face detection
    faces = face_cascade.detectMultiScale(img, 1.1, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx, cy = x + w//2, y + h//2
        results.append(("FACE", cx, cy))

    # Eye detection
    eyes = eye_cascade.detectMultiScale(img, 1.1, 5)
    for x, y, w, h in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx, cy = x + w//2, y + h//2
        results.append(("EYE", cx, cy))

    # Red object detection
    mask1 = cv2.inRange(imgHSV, lower_red1, upper_red1)
    mask2 = cv2.inRange(imgHSV, lower_red2, upper_red2)
    mask = mask1 + mask2
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelo)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelc)

    contours, _ = cv2.findContours(maskClose.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append(("OBJ", cx, cy))

    return img, results
