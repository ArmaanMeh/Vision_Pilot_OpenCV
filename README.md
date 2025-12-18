################################################################################
# OpenCV Pan and Tilt Object Tracker (Annotated Reference File)
################################################################################
# Introduction
#
# This project showcases a real-time object tracking system built using Python's
# OpenCV library for vision processing and PySerial for hardware communication.
# The system can operate in two primary modes: color-based tracking (Red Ball) 
# and feature-based tracking (Haar Cascade Face Detection). The system implements 
# a smooth, rate-limited Proportional (P) controller with integration to ensure 
# the target object remains centered in the camera's field of view. Command 
# signals are issued to a 2-DOF Pan-Tilt servo controller in a fixed format 
# (-1.0 to 1.0).
#
# The control flow uses a toggle-switch mechanism ('r', 'f') to dynamically 
# switch between tracking targets, offering flexible and adaptive behavior.
################################################################################

# ==============================================================================
# SECTION 1: IMPORTS, CONFIGURATION, AND GLOBAL STATE
# Purpose: Define external dependencies, hardware interfaces, and core constants.
# Key Rationale: Centralized settings allow for quick hardware/tuning adjustments 
#                and provide initial values for the control system.
# ==============================================================================

import cv2
import numpy as np
import serial
import time
import os 

# --- Hardware Settings ---
SERIAL_PORT = 'COM8'      # Target COM port for Arduino communication
BAUD_RATE = 9600
CAMERA_ID = 1             # Camera index (usually 0 or 1)

# --- Movement & Control Settings (P-Controller & Integrator) ---
SEND_RATE_HZ = 10          
SEND_INTERVAL = 1.0 / SEND_RATE_HZ 

DEAD_BAND_PIXELS = 35     # Error tolerance: no movement if error is within this range
SMOOTHING_FACTOR = 0.5    # Not actively used, placeholder
MAX_SPEED_CAP = 0.22      # Clamps the max change in position per frame (smooth motion limit)
P_GAIN = 0.0002           # Proportional constant: determines how aggressively the system responds to error
delta_pan = 0.0           # Stores instantaneous change required (velocity term)
delta_tilt = 0.0

# --- Detection Settings: Red Ball (HSV Thresholds) ---
LOWER_RED1 = np.array([0, 80, 85])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 80, 85])
UPPER_RED2 = np.array([180, 255, 255])

MIN_AREA = 1500           # Minimum object size to be considered a target
MAX_AREA = 60000         

# --- Haar Cascade Configuration ---
FACE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_eye.xml"
SMILE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_smile.xml"

# Global state variables for toggles
toggle_red_ball = False
toggle_face = False
toggle_eyes = False
toggle_smile = False

# Global variables for smoothing and serial communication
ser = None
last_send_time = 0
global_pan_position = 0.0 # Stores absolute integrated position (-1.0 to 1.0)
global_tilt_position = 0.0


# ==============================================================================
# SECTION 2: HARDWARE INTERFACE AND CONTROL FUNCTION
# Purpose: Manage serial connection and implement the core control logic (Integrator/Rate Limiter).
# Key Rationale: Integration of the velocity command (delta) into an absolute, 
#                clamped position ensures smooth servo movement.
# ==============================================================================

def init_serial():
    """Attempts to connect to the serial port without crashing."""
    global ser
    try:
        if ser is None or not ser.is_open:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) 
            print(f"Serial connected to {SERIAL_PORT}")
            time.sleep(2) # Wait for Arduino reset after connection
    except serial.SerialException:
        ser = None

def send_data_smoothed(d_pan, d_tilt):
    """
    Acts as the Integrator and Rate Limiter.
    Accumulates velocity (d_pan/d_tilt) into absolute position and sends it.
    """
    global ser, last_send_time, global_pan_position, global_tilt_position
    
    current_time = time.time()
    
    # Check rate limit before performing calculations
    if current_time - last_send_time < SEND_INTERVAL:
        return

    # 1. Update Global Position (Integration)
    global_pan_position += d_pan
    global_tilt_position += d_tilt
    
    # 2. Clamping (Saturation): Prevents position from exceeding the -1.0 to 1.0 limits
    global_pan_position = max(-1.0, min(1.0, global_pan_position))
    global_tilt_position = max(-1.0, min(1.0, global_tilt_position))

    # 3. Serial Communication
    if ser is None:
        init_serial()
        return
   
    # Message format: "pan_position,tilt_position\n" 
    msg = f"{global_pan_position:.4f},{global_tilt_position:.4f}\n" 

    try:
        ser.write(msg.encode('utf-8'))
        last_send_time = current_time
    except (serial.SerialException, OSError):
        ser = None


# ==============================================================================
# SECTION 3: INITIALIZATION AND CASCADE LOADING
# Purpose: Setup the camera stream, define image processing kernels, and load 
#          the pre-trained Haar Cascade XML files.
# Key Rationale: Cascades are loaded once at the start to ensure fast detection 
#                within the main loop. Error handling checks file paths.
# ==============================================================================

# Camera Setup
cam = cv2.VideoCapture(CAMERA_ID)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Image Processing Kernels for Morphology
kernel_open = np.ones((7, 7))
kernel_close = np.ones((11, 11))

# Load Haar Cascades
try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)
    
    if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
        raise FileNotFoundError
except FileNotFoundError:
    print("\n[ERROR] One or more Haar Cascade XML files not found.")
    print("Ensure the 'Haar_Cascades XML' folder is correct relative to the script.")
    exit()

WINDOW_NAME = "Tracker Control"
cv2.namedWindow(WINDOW_NAME)

print(f"\nSending {SEND_RATE_HZ} msgs/sec. | Dead-Band: {DEAD_BAND_PIXELS} pixels")
print("--- Keyboard Toggles ---")
print("Press 'r' for Red Ball | 'f' for Face (Control).")
print("Press 'e' for Eyes, 's' for Smile (Visual Only). Press 'q' to quit.")


# ==============================================================================
# SECTION 4: MAIN LOOP AND TRACKING LOGIC
# Purpose: The core execution loop that manages tracking, control calculation, 
#          and user interaction.
# Key Rationale: The loop prioritizes control calculation and then executes the 
#                `send_data_smoothed` function, which handles the rate limiting 
#                and serial transmission.
# ==============================================================================

# MAIN LOOP #
try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("Camera lost.")
            break

        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        
        # Reset control variables every frame
        delta_pan = 0.0 
        delta_tilt = 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        tracked_center_x = None
        face_rect = None

        # --- A. RED BALL TRACKING LOGIC (Color-based) ---
        if toggle_red_ball:
            # 1. Preprocessing: Blur for noise reduction, convert to HSV for color stability.
            blurred = cv2.GaussianBlur(img, (11, 11), 0)
            imgHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # 2. Masking and Morphology: Isolate red (dual mask), then clean the mask.
            mask1 = cv2.inRange(imgHSV, LOWER_RED1, UPPER_RED1)
            mask2 = cv2.inRange(imgHSV, LOWER_RED2, UPPER_RED2)
            mask = mask1 + mask2
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            # 3. Contour Finding: Locate the largest valid red object.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = None
            max_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if MIN_AREA < area < MAX_AREA:
                    x_bb, y_bb, cw_bb, ch_bb = cv2.boundingRect(cnt)
                    aspect = float(cw_bb) / ch_bb
                    if 0.5 < aspect < 1.5: 
                        if area > max_area:
                            max_area = area
                            largest_contour = cnt

            # 4. Centroid Calculation (Control Signal Source)
            if largest_contour is not None:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    tracked_center_x, tracked_center_y = cx, cy

                # Visuals for Ball
                x_bb, y_bb, cw_bb, ch_bb = cv2.boundingRect(largest_contour)
                cv2.rectangle(img, (x_bb, y_bb), (x_bb+cw_bb, y_bb+ch_bb), (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)


        # --- B. FACE TRACKING LOGIC (Feature-based) ---
        elif toggle_face: 
            
            # 1. Detection: Find faces in the grayscale image.
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            # 2. Select Largest Face: Prioritize tracking the closest/largest face.
            largest_face = None
            max_face_area = 0
            for rect in faces:
                x, y, w, h = rect
                area = w * h
                if area > max_face_area:
                    max_face_area = area
                    largest_face = rect

            if largest_face is not None:
                x, y, w, h = largest_face
                face_rect = largest_face # Stored for eyes/smile sub-detection
                
                # 3. Calculate Center (Control Signal Source)
                cx = x + w // 2
                cy = y + h // 2
                tracked_center_x, tracked_center_y = cx, cy

                # Draw Face rectangle
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, "Tracking Face", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

        
        # --- C. PROPORTIONAL (P) CONTROL CALCULATION ---
        if tracked_center_x is not None:
            # 1. Calculate Error (in pixels)
            error_x = tracked_center_x - center_x
            error_y = tracked_center_y - center_y

            pan_direction = -1 
            tilt_direction = -1  
           
            # 2. Apply Dead-Band and P-Gain to get Delta (Velocity)
            if abs(error_x) > DEAD_BAND_PIXELS:
                delta_pan = error_x * P_GAIN * pan_direction
            
            if abs(error_y) > DEAD_BAND_PIXELS:
                delta_tilt = error_y * P_GAIN * tilt_direction

            # 3. Apply Max Speed Cap
            delta_pan = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, delta_pan)) 
            delta_tilt = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, delta_tilt))
            
        
        # --- D. VISUAL OVERLAYS (Eyes/Smile) ---
        # Checks if a face was detected either for control or just for visual toggles
        if face_rect is None and (toggle_eyes or toggle_smile):
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            if len(faces) > 0:
                max_area = 0
                for rect in faces:
                    x, y, w, h = rect
                    area = w * h
                    if area > max_area:
                        max_area = area
                        face_rect = rect

        if face_rect is not None:
            x, y, w, h = face_rect
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            if toggle_eyes:
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

            if toggle_smile:
                smile = smile_cascade.detectMultiScale(roi_gray, 
                                                         scaleFactor=1.7, 
                                                         minNeighbors=22, 
                                                         minSize=(25, 25))
                for (sx, sy, sw, sh) in smile:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2)


        # --- E. Execute Control Command and Display ---
        # The core hardware communication step
        send_data_smoothed(delta_pan, delta_tilt)
            
        # UI Overlay (Status Text and Box)
        cv2.putText(img, f"Pan POS: {global_pan_position:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Tilt POS: {global_tilt_position:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.rectangle(img, (center_x - DEAD_BAND_PIXELS, center_y - DEAD_BAND_PIXELS), 
                      (center_x + DEAD_BAND_PIXELS, center_y + DEAD_BAND_PIXELS), (0, 255, 255), 1)

        # Toggle Status Indicators (bottom right)
        y_pos = h - 20
        cv2.putText(img, f"R: {'ON' if toggle_red_ball else 'OFF'}", (w - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_red_ball else (0, 0, 255), 2)
        cv2.putText(img, f"F: {'ON' if toggle_face else 'OFF'}", (w - 150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_face else (0, 0, 255), 2)
        cv2.putText(img, f"E: {'ON' if toggle_eyes else 'OFF'}", (w - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_eyes else (0, 0, 255), 2)
        cv2.putText(img, f"S: {'ON' if toggle_smile else 'OFF'}", (w - 50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_smile else (0, 0, 255), 2)


        # --- F. KEYBOARD INPUT HANDLING (Toggle Switch Logic) ---
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            toggle_red_ball = not toggle_red_ball
            if toggle_red_ball: 
                toggle_face = False 
                print("Red Ball TRACKING (Control) is now ON (Face Control OFF).")
            else:
                print("Red Ball TRACKING (Control) is now OFF.")

        elif key == ord('f'):
            toggle_face = not toggle_face
            if toggle_face: 
                toggle_red_ball = False 
                print(f"Face TRACKING (Control) is now ON (Red Ball OFF).")
            else:
                print(f"Face TRACKING (Control) is now OFF.")

        elif key == ord('e'):
            toggle_eyes = not toggle_eyes
            print(f"Eye detection is now {'ON' if toggle_eyes else 'OFF'}")
            
        elif key == ord('s'):
            toggle_smile = not toggle_smile
            print(f"Smile detection is now {'ON' if toggle_smile else 'OFF'}")

        cv2.imshow(WINDOW_NAME, img)
        
except KeyboardInterrupt:
    pass
finally:
    # --- G. EXIT AND CLEANUP ---
    # Crucial step: release hardware resources gracefully.
    cam.release()
    if ser is not None:
        ser.close()
    cv2.destroyAllWindows()

# ==============================================================================
# END OF SCRIPT
# ==============================================================================