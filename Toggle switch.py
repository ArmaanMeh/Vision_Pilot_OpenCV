import cv2
import numpy as np
import serial
import time
import collections

# Serial Port 
SERIAL_PORT = 'COM3' 
BAUD_RATE = 9600

# Hardware Servo Limits
PAN_MIN = 0
PAN_MAX = 170
TILT_MIN = 0
TILT_MAX = 120

# Tracking Sensitivity (Higher = Faster movement)
KP_PAN = 1.5
KP_TILT = 1.5
KI_PAN = 0.05
KI_TILT = 0.05  
FILTER_SIZE = 5

# SERIAL SETUP

try:
    arduino_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Wait for Arduino to reset
    print(f"Connected to {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Error connecting to serial: {e}")
    arduino_serial = None

# Initial Angles
current_pan = 90.0
current_tilt = 60.0

# PI Controller State
global integral_pan, integral_tilt
integral_pan = 0.0
integral_tilt = 0.0
MAX_INTEGRAL = 20.0

pan_history = collections.deque([current_pan]*FILTER_SIZE, maxlen=FILTER_SIZE)
tilt_history = collections.deque([current_tilt]*FILTER_SIZE, maxlen=FILTER_SIZE)


def send_pan_tilt(pan, tilt):
    """
    Clamps angles to limits and sends to Arduino via Serial.
    """
    # Clamp values to hardware limits
    pan = max(PAN_MIN, min(PAN_MAX, int(pan)))
    tilt = max(TILT_MIN, min(TILT_MAX, int(tilt)))
    
    if arduino_serial:
        # Format: PxxxTxxx\n
        command = f"P{pan:03d}T{tilt:03d}\n"
        try:
            arduino_serial.write(command.encode('utf-8'))
        except Exception as e:
            print(f"Serial Write Error: {e}")

# ==========================================
# CV SETUP
# ==========================================
# Load Haar Cascades
try:
    face_cascade = cv2.CascadeClassifier("Haar_Cascades XML\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("Haar_Cascades XML\haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier("Haar_Cascades XML\haarcascade_smile.xml")
except Exception as e:
    print("Error loading cascades. Check your file paths!")

# HSV Ranges for Red
lower_red1 = np.array([0, 80, 60])
upper_red1 = np.array([10, 255, 255]) 
lower_red2 = np.array([170, 80, 60])
upper_red2 = np.array([180, 255, 255]) 

# Morphological kernels
kernelo = np.ones((5, 5))
kernelc = np.ones((10, 10)) 

# Camera Setup
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Flags
show_red = True
show_face = False  
show_eye = False
show_smile = False

# Send Initial Position
send_pan_tilt(current_pan, current_tilt)

print("\nCONTROLS:")
print("'f' - Toggle Face Detection")
print("'r' - Toggle Red Object Detection")
print("'e' - Toggle Eye Detection")
print("'s' - Toggle Smile Detection")
print("'q' - Quit")

while True:
    ret, img = cam.read()
    if not ret: break

    # Get dimensions for normalized tracking
    h, w, c = img.shape
    center_x, center_y = w // 2, h // 2
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # We need a target to track (x, y)
    target_x = None
    target_y = None
    target_label = "No Target"
    
    # 1. FACE DETECTION 
    if show_face:
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        
        # If faces found, pick the largest one to track
        if len(faces) > 0:
            (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
            cv2.rectangle(img, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
            cv2.putText(img, "Face", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
            
            target_x = x + fw // 2
            target_y = y + fh // 2
            target_label = "Tracking Face"

    # 2. EYE DETECTION (Visual only, usually too small to track stably)
    if show_eye:
        eyes = eye_cascade.detectMultiScale(img, 1.1, 5)
        for x, y, ew, eh in eyes:
            cv2.rectangle(img, (x, y), (x+ew, y+eh), (255, 255, 0), 2)

    # 3. SMILE DETECTION (Visual only)
    if show_smile:
        smiles = smile_cascade.detectMultiScale(img, 1.8, 20)
        for x, y, sw, sh in smiles:
            cv2.rectangle(img, (x, y), (x+sw, y+sh), (0, 255, 255), 2)

    # 4. RED OBJECT DETECTION 
    if show_red:
        mask1 = cv2.inRange(imgHSV, lower_red1, upper_red1)
        mask2 = cv2.inRange(imgHSV, lower_red2, upper_red2)
        mask = mask1 + mask2
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelo)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelc)

        contours, hierarchy = cv2.findContours(maskClose.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all red objects
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # Find largest red object
        largest_contour = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500: # Filter noise
                if area > max_area:
                    max_area = area
                    largest_contour = cnt
                
                # Draw box around all
                x, y, cw, ch = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+cw, y+ch), (255, 0, 0), 1)

        # If we found a red object AND we aren't tracking a face
        if largest_contour is not None and target_x is None:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                target_x = cx
                target_y = cy
                target_label = "Tracking Red"
                cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)

    # ==========================================
    # TRACKING LOGIC (-1 to 1 Integration)
    # ==========================================
        # Integral Component (Accumulates small, persistent error)
        
        # Only perform tracking logic if target_x and target_y are not None
        if target_x is not None and target_y is not None:
            # Calculate Normalized Error (-1.0 to 1.0)
            error_pan = (target_x - center_x) / (w / 2)
            error_tilt = (target_y - center_y) / (h / 2)

            integral_pan += error_pan
            integral_tilt += error_tilt

            # Anti-windup (Clamping the integral component)
            integral_pan = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, integral_pan))
            integral_tilt = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, integral_tilt))

            # PI Control Formula: Output = Kp * Error + Ki * Integral [Image of PI Controller Block Diagram]
            pan_correction = (KP_PAN * error_pan) + (KI_PAN * integral_pan)
            tilt_correction = (KP_TILT * error_tilt) + (KI_TILT * integral_tilt)

            # Apply correction to current angle
            current_pan += pan_correction
            current_tilt -= tilt_correction
            
            # Angle Smoothing: Add to history
            pan_history.append(current_pan)
            tilt_history.append(current_tilt)
            
            # Angle Smoothing: Calculate filtered angle (Moving Average)
            smooth_pan = sum(pan_history) / FILTER_SIZE
            smooth_tilt = sum(tilt_history) / FILTER_SIZE
            # Send to Arduino (Function handles clamping)
            send_pan_tilt(smooth_pan, smooth_tilt)

            # Visuals
            cv2.line(img, (center_x, center_y), (target_x, target_y), (0, 0, 0), 2)
            cv2.putText(img, f"{target_label} | Pan:{int(current_pan)} Tilt:{int(current_tilt)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(img, f"{target_label}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Tracking System", img)

    # Keys
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'): show_red = not show_red; print(f"Red: {show_red}")
    elif key == ord('f'): show_face = not show_face; print(f"Face: {show_face}")
    elif key == ord('e'): show_eye = not show_eye; print(f"Eye: {show_eye}")
    elif key == ord('s'): show_smile = not show_smile; print(f"Smile: {show_smile}")

cam.release()
cv2.destroyAllWindows()
if arduino_serial: arduino_serial.close()