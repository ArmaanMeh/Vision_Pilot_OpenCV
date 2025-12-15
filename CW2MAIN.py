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
SEND_RATE_HZ = 1          
SEND_INTERVAL = 1.0 / SEND_RATE_HZ 

DEAD_BAND_PIXELS = 35 
SMOOTHING_FACTOR = 0.5    

# --- Detection Settings ---
LOWER_RED1 = np.array([0, 80, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 80, 70])
UPPER_RED2 = np.array([180, 255, 255])

MIN_AREA = 800            
MAX_AREA = 130000         
# ======================================================

# --- Haar Cascade Configuration (Using the paths from your sample code) ---
FACE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_eye.xml"
SMILE_CASCADE_PATH = "Haar_Cascades XML/haarcascade_smile.xml"

# Global state variables for toggles
toggle_face = False
toggle_eyes = False
toggle_smile = False

# Global variables for smoothing
ser = None
last_send_time = 0
prev_pan = 0.0
prev_tilt = 0.0

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

def send_data_smoothed(target_pan, target_tilt):
    """Applies smoothing, prints, and sends data at a fixed interval."""
    global ser, last_send_time, prev_pan, prev_tilt
    
    # 1. Apply Exponential Smoothing
    smooth_pan = (prev_pan * (1 - SMOOTHING_FACTOR)) + (target_pan * SMOOTHING_FACTOR)
    smooth_tilt = (prev_tilt * (1 - SMOOTHING_FACTOR)) + (target_tilt * SMOOTHING_FACTOR)
    
    prev_pan = smooth_pan
    prev_tilt = smooth_tilt

    current_time = time.time()
    if current_time - last_send_time < SEND_INTERVAL:
        return

    if ser is None:
        init_serial()
        return

    msg = f"P{smooth_pan:.2f},T{smooth_tilt:.2f}\n"
    print(f"[SERIAL OUT] Sending: {msg.strip()}") 

    try:
        ser.write(msg.encode('utf-8'))
        last_send_time = current_time
    except (serial.SerialException, OSError):
        ser = None 

# --- MAIN SETUP ---
cam = cv2.VideoCapture(CAMERA_ID)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

kernel_open = np.ones((5, 5))
kernel_close = np.ones((10, 10))

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

WINDOW_NAME = "Red Ball Tracker"
cv2.namedWindow(WINDOW_NAME)

print(f"\n[SYSTEM READY] Sending {SEND_RATE_HZ} msgs/sec.")
print(f"[CONFIG] Dead-Band: {DEAD_BAND_PIXELS} pixels | Smoothing: {SMOOTHING_FACTOR}")
print("--- Keyboard Toggles ---")
print("Press 'f' to toggle Face, 'e' for Eyes (requires Face ON), 's' for Smile (requires Face ON). Press 'q' to quit.")

try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Camera lost.")
            break

        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        
        raw_pan = 0.0
        raw_tilt = 0.0
        
        # Convert image to grayscale for Haar Cascades (more efficient)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # ======================================================
        # 1. RED BALL TRACKING LOGIC (PRIMARY CONTROL)
        # ======================================================
        
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
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = float(cw) / ch
                if 0.5 < aspect < 1.5: 
                    if area > max_area:
                        max_area = area
                        largest_contour = cnt
                        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

        # Calculate Control Signals if ball is found
        if largest_contour is not None:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x_bb, y_bb, cw_bb, ch_bb = cv2.boundingRect(largest_contour)
                cx, cy = x_bb + cw_bb//2, y_bb + ch_bb//2

            # Visuals for Ball
            x_bb, y_bb, cw_bb, ch_bb = cv2.boundingRect(largest_contour)
            cv2.rectangle(img, (x_bb, y_bb), (x_bb+cw_bb, y_bb+ch_bb), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(img, (center_x, center_y), (cx, cy), (0, 255, 255), 2)

            # --- CALCULATE CONTROL ---
            error_x = cx - center_x
            error_y = cy - center_y
            
            if abs(error_x) > DEAD_BAND_PIXELS:
                raw_pan = -1 * (error_x / (w / 2.0))
            
            if abs(error_y) > DEAD_BAND_PIXELS:
                raw_tilt = -error_y / (h / 2.0) 

            raw_pan = max(-1.0, min(1.0, raw_pan))
            raw_tilt = max(-1.0, min(1.0, raw_tilt))
            
            send_data_smoothed(raw_pan, raw_tilt)
            
            cv2.putText(img, f"Pan: {prev_pan:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Tilt: {prev_tilt:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        else:
            send_data_smoothed(0.0, 0.0)
            
        # ======================================================
        # 2. FACE/EYE/SMILE DETECTION (VISUAL ONLY)
        #    Integrated the structure from your provided code
        # ======================================================
        
        if toggle_face:
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5) # Use gray image for detection
            
            for (x, y, w, h) in faces:
                # Draw Face rectangle and text
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # Use Blue for Face
                cv2.putText(img, "Face detected", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                
                # Define ROI for Eye and Smile detection inside the face box
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                if toggle_eyes:
                    # Detect eyes within the face ROI
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                    for (ex, ey, ew, eh) in eyes:
                        # Draw rectangle on the color ROI
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2) # Yellow for Eyes
                        cv2.putText(roi_color, "Eyes", (ex, ey-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

                if toggle_smile:
                    # Detect smile within the face ROI
                    smile = smile_cascade.detectMultiScale(roi_gray, 
                                                           scaleFactor=1.7, 
                                                           minNeighbors=22, 
                                                           minSize=(25, 25))
                    for (sx, sy, sw, sh) in smile:
                        # Draw rectangle on the color ROI
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2) # Magenta for Smile
                        cv2.putText(roi_color, "Smile", (sx, sy + sh + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)


        # UI Overlay - Status Indicators
        
        # Dead-band zone
        cv2.rectangle(img, 
                      (center_x - DEAD_BAND_PIXELS, center_y - DEAD_BAND_PIXELS), 
                      (center_x + DEAD_BAND_PIXELS, center_y + DEAD_BAND_PIXELS), 
                      (0, 255, 255), 1)

        # Serial Connection Status
        status = "CONN" if ser is not None else "DISC"
        col = (0, 255, 0) if ser is not None else (0, 0, 255)
        cv2.putText(img, status, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        
        # Toggle Status Indicators
        y_pos = h - 20
        cv2.putText(img, f"F: {'ON' if toggle_face else 'OFF'}", (w - 150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_face else (0, 0, 255), 2)
        cv2.putText(img, f"E: {'ON' if toggle_eyes else 'OFF'}", (w - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_eyes else (0, 0, 255), 2)
        cv2.putText(img, f"S: {'ON' if toggle_smile else 'OFF'}", (w - 50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if toggle_smile else (0, 0, 255), 2)


        # --- KEYBOARD INPUT HANDLING ---
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('f'):
            toggle_face = not toggle_face
            # Turn off sub-detections if face is toggled off
            if not toggle_face: 
                toggle_eyes = False
                toggle_smile = False 
            print(f"[TOGGLE] Face detection is now {'ON' if toggle_face else 'OFF'}")
        elif key == ord('e'):
            if toggle_face:
                toggle_eyes = not toggle_eyes
                print(f"[TOGGLE] Eye detection is now {'ON' if toggle_eyes else 'OFF'}")
            else:
                print("[WARNING] Face detection must be ON to toggle Eyes.")
        elif key == ord('s'):
            if toggle_face:
                toggle_smile = not toggle_smile
                print(f"[TOGGLE] Smile detection is now {'ON' if toggle_smile else 'OFF'}")
            else:
                print("[WARNING] Face detection must be ON to toggle Smile.")


        cv2.imshow(WINDOW_NAME, img)

except KeyboardInterrupt:
    pass
finally:
    cam.release()
    if ser is not None:
        ser.close()
    cv2.destroyAllWindows()