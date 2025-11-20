import cv2
import numpy as np
#Define HSV ranges for red color
lower_red1 = np.array([0, 80, 60])
upper_red1 = np.array([10, 255, 255]) 
lower_red2 = np.array([170, 80, 60])
upper_red2 = np.array([180, 255, 255]) 

# Define morphological operation kernels
kernelo = np.ones((5, 5))
kernelc = np.ones((10, 10)) 

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
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
    cv2.imshow("Mask", mask)
    cv2.imshow("Mask Open", maskOpen)
    cv2.imshow("Mask Close", maskClose)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
