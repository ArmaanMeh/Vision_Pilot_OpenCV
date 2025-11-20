import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break
 # Find contours
    contours,heriarchy = cv2.findContours(maskClose.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and bounding boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(img,contours,-1,(0,255,0),2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(img, ".object detected", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 1)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(img, f"Centroid: ({cx},{cy})", (cx + 30, cy - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)


    cv2.imshow("Original", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
