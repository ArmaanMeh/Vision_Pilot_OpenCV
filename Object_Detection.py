import cv2
import numpy as np

cam = cv2.VideoCapture(0)
# Auto exposure on
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

while True:
    ret, img = cam.read()
    if not ret:
        break

    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Mask Open", maskOpen)
    cv2.imshow("Mask Close", maskClose)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
