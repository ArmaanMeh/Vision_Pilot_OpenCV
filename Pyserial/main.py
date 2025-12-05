import cv2
from detection import detect_objects
from serial_utils import SerialComm

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ser = SerialComm(port="COM3", baudrate=9600)

    while True:
        ret, img = cam.read()
        if not ret:
            break

        img, detections = detect_objects(img)

        # Send detections to Arduino
        for tag, cx, cy in detections:
            ser.send(tag, cx, cy)

        cv2.imshow("Frame", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()
