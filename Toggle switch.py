import cv2
import numpy as np

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier("opencvCW\\OPENCV CW\\Haar_Cadcades\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("opencvCW\\OPENCV CW\\Haar_Cadcades\\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("opencvCW\\OPENCV CW\\Haar_Cadcades\\haarcascade_smile.xml")

# Define HSV ranges for red color
lower_red1 = np.array([0, 80, 60])
upper_red1 = np.array([10, 255, 255]) 
lower_red2 = np.array([170, 80, 60])
upper_red2 = np.array([180, 255, 255]) 

# Define morphological operation kernels
kernelo = np.ones((5, 5))
kernelc = np.ones((10, 10)) 

# Initialize video capture
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Toggle flags
show_red = False
show_face = False
show_eye = False
show_smile = False

while True:
    ret, img = cam.read()
    if not ret:
        break

    # Convert to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Face Detection
    if show_face:
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Face detected", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

    # Eye Detection
    if show_eye:
        eyes = eye_cascade.detectMultiScale(img, 1.1, 5)
        for x, y, w, h in eyes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Eyes detected", (x-40, y-40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

    # Smile Detection
    if show_smile:
        smiles = smile_cascade.detectMultiScale(img, 1.1, 5)
        for x, y, w, h in smiles:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Smile detected", (x-40, y-40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

    # Red Object Detection
    if show_red:
        mask1 = cv2.inRange(imgHSV, lower_red1, upper_red1)
        mask2 = cv2.inRange(imgHSV, lower_red2, upper_red2)
        mask = mask1 + mask2

        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelo)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelc)

        contours, hierarchy = cv2.findContours(maskClose.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
                cv2.putText(img, "Object detected", (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 1)

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(img, f"Centroid: ({cx},{cy})", (cx+30, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        cv2.imshow("Mask", mask)
        cv2.imshow("Mask Open", maskOpen)
        cv2.imshow("Mask Close", maskClose)

    cv2.imshow("Original", img)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        show_red = not show_red
        print("Red detection:", show_red)
    elif key == ord('f'):
        show_face = not show_face
        print("Face detection:", show_face)
    elif key == ord('e'):
        show_eye = not show_eye
        print("Eye detection:", show_eye)
    elif key == ord('s'):
        show_smile = not show_smile
        print("Smile detection:", show_smile)

cam.release()
cv2.destroyAllWindows()
