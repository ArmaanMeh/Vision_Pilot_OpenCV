import cv2
import numpy as np
import serial
import time
import os 

# ================= USER CONFIGURATION =================

# --- Hardware Settings ---
SERIAL_PORT = 'COM8' 
BAUD_RATE = 9600
CAMERA_ID = 1

# --- Movement Settings ---
SEND_RATE_HZ = 10 
SEND_INTERVAL = 1.0 / SEND_RATE_HZ 

DEAD_BAND_PIXELS = 50 
MAX_SPEED_CAP = 0.25
P_GAIN = 0.00021 

# --- Color Detection Settings (Red Ball) ---
LOWER_RED1 = np.array([0, 80, 85])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 80, 85])
UPPER_RED2 = np.array([180, 255, 255])
MIN_AREA = 1500 
MAX_AREA = 60000

# --- Haar Cascade Configuration ---
FACE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_eye.xml"
SMILE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_smile.xml"

# ================= GLOBAL STATE =================

# Global state variables for toggles
toggle_red_ball = False
toggle_face = False
toggle_eyes = False
toggle_smile = False

# Global variables for smoothing and serial
ser = None
last_send_time = 0
global_pan_position = 0.0
global_tilt_position = 0.0

# ================= FUNCTIONS =================

def init_serial():
    """Attempts to connect to the serial port without crashing."""
    global ser
    try:
        if ser is None or not ser.is_open:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) 
            print(f"[INFO] Serial connected to {SERIAL_PORT}")
            time.sleep(2) 
    except serial.SerialException:
        ser = None

def send_data_smoothed(d_pan, d_tilt):
    """
    Integrator (I-term) for control. Accumulates velocity (d_pan/d_tilt)
    into absolute position and sends the command.
    """
    global ser, last_send_time, global_pan_position, global_tilt_position
    
    current_time = time.time()
    
    # Rate limit check
    if current_time - last_send_time < SEND_INTERVAL:
        return

    # 1. Update Global Position (Integration)
    global_pan_position += d_pan
    global_tilt_position += d_tilt
    
    # Clamping positions to the valid range [-1.0, 1.0]
    global_pan_position = max(-1.0, min(1.0, global_pan_position))
    global_tilt_position = max(-1.0, min(1.0, global_tilt_position))

    # 2. Serial Communication
    if ser is None:
        init_serial()
        return
    
    msg = f"{global_pan_position:.4f},{global_tilt_position:.4f}\n" 

    try:
        ser.write(msg.encode('utf-8'))
        last_send_time = current_time
    except (serial.SerialException, OSError):
        ser = None

# ================= MAIN SETUP =================

cam = cv2.VideoCapture(CAMERA_ID)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

kernel_open = np.ones((7, 7))
kernel_close = np.ones((11, 11))

# Load Haar Cascades using the specified file paths
try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)
    
    if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
        raise FileNotFoundError
except FileNotFoundError:
    print("\n[CRITICAL ERROR] One or more Haar Cascade XML files not found.")
    print(f"Check your paths: {FACE_CASCADE_PATH}, {EYE_CASCADE_PATH}, {SMILE_CASCADE_PATH}")
    print("Ensure the 'Haar_Cascades XML' folder is correct relative to the script.")
    exit()

WINDOW_NAME = "Tracker Control"
cv2.namedWindow(WINDOW_NAME)

print(f"\n[SYSTEM READY] Sending {SEND_RATE_HZ} msgs/sec.")
print(f"[CONFIG] Dead-Band: {DEAD_BAND_PIXELS} pixels | P_GAIN: {P_GAIN} | Max Speed: {MAX_SPEED_CAP}")
print("--- Keyboard Toggles ---")
print("Press 'r' to toggle Red Ball Tracking (Servo Control).")
print("Press 'f' to toggle Face Tracking (Servo Control).")
print("Press 'e' for Eyes, 's' for Smile (Visual Only - Requires face detected). Press 'q' to quit.")

# ================= MAIN LOOP =================

try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Camera lost.")
            break

        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        
        # Control signals (will remain 0.0 unless a tracking mode is active)
        delta_pan = 0.0
        delta_tilt = 0.0
        display_color = (255, 0, 0) # Default: Blue (Inactive)

        # Convert image to grayscale once for all Haar Cascades
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        # Initialize tracked object details
        tracked_center_x = None
        tracked_center_y = None
        face_rect = None # To hold the largest detected face coordinates

        # --- 1. RED BALL TRACKING LOGIC (Exclusive Control Mode) ---
        if toggle_red_ball:
            display_color = (0, 255, 0) # Green (Red Ball Active)

            # Image Pre-processing for Color Tracking
            blurred = cv2.GaussianBlur(img, (11, 11), 0)
            imgHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Red Mask Generation and Cleanup
            mask1 = cv2.inRange(imgHSV, LOWER_RED1, UPPER_RED1)
            mask2 = cv2.inRange(imgHSV, LOWER_RED2, UPPER_RED2)
            mask = mask1 + mask2
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            # Find Contours and find largest valid Ball
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
                            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

            # Calculate Control Signals if ball is found
            if largest_contour is not None:
                M = cv2.moments(largest_contour)
                cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else center_x
                cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else center_y

                # Store center for control
                tracked_center_x, tracked_center_y = cx, cy

                # Visuals for Ball
                x_bb, y_bb, cw_bb, ch_bb = cv2.boundingRect(largest_contour)
                cv2.rectangle(img, (x_bb, y_bb), (x_bb+cw_bb, y_bb+ch_bb), (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(img, (center_x, center_y), (cx, cy), (0, 255, 255), 2)

        
        # --- 2. FACE DETECTION & TRACKING LOGIC (Exclusive Control Mode) ---
        elif toggle_face: 
            display_color = (0, 0, 255) # Red (Face Active)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            largest_face = None
            max_face_area = 0

            # Find the largest face to track
            for rect in faces:
                x, y, w, h = rect
                area = w * h
                if area > max_face_area:
                    max_face_area = area
                    largest_face = rect

            if largest_face is not None:
                x, y, w, h = largest_face
                face_rect = largest_face # Store face rect for sub-detections later
                
                # Calculate center of the tracked face
                cx = x + w // 2
                cy = y + h // 2
                
                # Store center for control
                tracked_center_x, tracked_center_y = cx, cy

                # Draw Face rectangle and text (Control Active)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, "Tracking Face", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                cv2.circle(img, (cx, cy), 5, (255, 255, 0), -1)
        
        # --- END OF EXCLUSIVE TRACKING MODES ---
        
        # --- 3. COMMON CONTROL CALCULATION ---
        # Only calculate control signals if a target was successfully found by the active mode
        if tracked_center_x is not None:
            # --- CALCULATE P-CONTROL ---
            error_x = tracked_center_x - center_x
            error_y = tracked_center_y - center_y
            
            pan_direction = -1 
            tilt_direction = -1 
            
            if abs(error_x) > DEAD_BAND_PIXELS:
                delta_pan = error_x * P_GAIN * pan_direction
            
            if abs(error_y) > DEAD_BAND_PIXELS:
                delta_tilt = error_y * P_GAIN * tilt_direction

            # Clamp speed
            delta_pan = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, delta_pan))
            delta_tilt = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, delta_tilt))


        # --- 4. INDEPENDENT VISUAL SUB-DETECTIONS (Eyes and Smile) ---
        # This section runs independently of whether Face Tracking is controlling the servo.
        # It only needs a face to be detected.

        # If Face Tracking wasn't active, we need to detect a face for Eyes/Smile visuals
        if face_rect is None and (toggle_eyes or toggle_smile):
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            if len(faces) > 0:
                # Find largest face for sub-detection drawing
                max_area = 0
                for rect in faces:
                    x, y, w, h = rect
                    area = w * h
                    if area > max_area:
                        max_area = area
                        face_rect = rect
                        # Draw a simple outline if Face Control is OFF
                        if not toggle_face:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)

        if face_rect is not None:
            x, y, w, h = face_rect
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Eye Detection (Visual Only)
            if toggle_eyes:
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2) # Yellow
                    cv2.putText(roi_color, "Eyes", (ex, ey-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

            # Smile Detection (Visual Only)
            if toggle_smile:
                smile = smile_cascade.detectMultiScale(roi_gray, 
                                                         scaleFactor=1.7, 
                                                         minNeighbors=22, 
                                                         minSize=(25, 25))
                for (sx, sy, sw, sh) in smile:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2) # Magenta
                    cv2.putText(roi_color, "Smile", (sx, sy + sh + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

        
        # --- 5. SERVO COMMAND & UI UPDATES ---
        
        # Send the calculated delta command (0.0 if no object/face was tracked)
        send_data_smoothed(delta_pan, delta_tilt)

        # Update display text to show the ACTUAL global position
        cv2.putText(img, f"Pan POS: {global_pan_position:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
        cv2.putText(img, f"Tilt POS: {global_tilt_position:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)


        # UI Overlay - Status Indicators
        
        # Dead-band zone visual
        cv2.rectangle(img, 
                      (center_x - DEAD_BAND_PIXELS, center_y - DEAD_BAND_PIXELS), 
                      (center_x + DEAD_BAND_PIXELS, center_y + DEAD_BAND_PIXELS), 
                      (0, 255, 255), 1)

        # Serial Connection Status
        status = "CONN" if ser is not None else "DISC"
        col = (0, 255, 0) if ser is not None else (0, 0, 255)
        cv2.putText(img, status, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        
        # Toggle Status Indicators (R: Red Ball, F: Face, E: Eyes, S: Smile)
        y_pos = h - 20
        # Red Ball Tracker Status
        cv2.putText(img, f"R: {'ON' if toggle_red_ball else 'OFF'}", (w - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_red_ball else (0, 0, 255), 2)
        # Face/Eye/Smile Status
        is_face_active_control = toggle_face and not toggle_red_ball
        cv2.putText(img, f"F: {'ON' if is_face_active_control else 'OFF'}", (w - 150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_face_active_control else (0, 0, 255), 2)
        cv2.putText(img, f"E: {'ON' if toggle_eyes else 'OFF'}", (w - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if toggle_eyes else (0, 0, 255), 2)
        cv2.putText(img, f"S: {'ON' if toggle_smile else 'OFF'}", (w - 50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255) if toggle_smile else (0, 0, 255), 2)


        # --- KEYBOARD INPUT HANDLING (Toggle Switches) ---
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            # TOGGLE RED BALL TRACKING (Servo Control)
            toggle_red_ball = not toggle_red_ball
            if toggle_red_ball:
                toggle_face = False 
                print("[TOGGLE] Red Ball TRACKING (Control) is now ON (Face Control OFF).")
            else:
                print("[TOGGLE] Red Ball TRACKING (Control) is now OFF.")

        elif key == ord('f'):
            # TOGGLE FACE TRACKING (Servo Control)
            toggle_face = not toggle_face
            if toggle_face:
                toggle_red_ball = False 
                print(f"[TOGGLE] Face TRACKING (Control) is now ON (Red Ball OFF).")
            else:
                print(f"[TOGGLE] Face TRACKING (Control) is now OFF.")
            
        elif key == ord('e'):
            # TOGGLE EYES (Visual Only)
            toggle_eyes = not toggle_eyes
            print(f"[TOGGLE] Eye detection is now {'ON' if toggle_eyes else 'OFF'}")
            
        elif key == ord('s'):
            # TOGGLE SMILE (Visual Only)
            toggle_smile = not toggle_smile
            print(f"[TOGGLE] Smile detection is now {'ON' if toggle_smile else 'OFF'}")


        cv2.imshow(WINDOW_NAME, img)

# ================= CLEANUP =================
except KeyboardInterrupt:
    pass
finally:
    # Release camera and serial port
    cam.release()
    if ser is not None:
        ser.close()
    cv2.destroyAllWindows()