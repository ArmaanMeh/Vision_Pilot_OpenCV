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
    ret, img = cap.read()
    if not ret:
        break
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(imghsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(imghsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Morphological operations
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelo)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelc)

     
                
    cv2.imshow("Original", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
