import cv2
import numpy as np
import serial # Import the serial library
import time # For a short delay after opening the serial port

# --- Serial Communication Setup ---
# **IMPORTANT:**
# 1. Change 'COM3' to the port your Arduino is connected to (e.g., 'COM3' on Windows, '/dev/ttyACM0' or '/dev/ttyUSB0' on Linux/Mac).
# 2. Ensure the baud rate (9600) matches the rate set in your Arduino sketch.
try:
    # Initialize serial port connection
    arduino_serial = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2) # Wait for the serial port to initialize
    print("Serial port opened successfully.")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    arduino_serial = None # Set to None if connection fails

def send_pan_tilt(pan_angle, tilt_angle):
    """
    Sends Pan and Tilt angles to the Arduino via serial communication.
    Angles should be between 0 and 180.
    The message format is: "P<pan_angle>T<tilt_angle>\n"
    """
    if arduino_serial is None:
        print("Serial port is not connected. Cannot send data.")
        return

    # Ensure angles are within 0-180 range (for standard servo motors)
    pan_angle = max(0, min(180, int(pan_angle)))
    tilt_angle = max(0, min(180, int(tilt_angle)))

    # Create the command string
    command = f"P{pan_angle:03d}T{tilt_angle:03d}\n"
    
    try:
        arduino_serial.write(command.encode('utf-8'))
        # print(f"Sent: {command.strip()}") # Uncomment for debugging
    except Exception as e:
        print(f"Error sending serial data: {e}")

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier("Haar_Cascades XML\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("Haar_Cascades XML\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("Haar_Cascades XML\haarcascade_smile.xml")

# Define HSV ranges for red color
lower_red1 = np.array([0, 80, 60])
upper_red1 = np.array([10, 255, 255]) 
lower_red2 = np.array([170, 80, 60])
upper_red2 = np.array([180, 255, 255]) 

# Define morphological operation kernels
kernelo = np.ones((5, 5))
kernelc = np.ones((10, 10)) 

# Initialize video capture
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get frame dimensions for centering calculations
# Note: This should be done after cam.read() for better robustness
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width // 2
center_y = frame_height // 2

# Initial Servo Angles (assuming 90 degrees is the center position)
current_pan = 90
current_tilt = 90
send_pan_tilt(current_pan, current_tilt) # Send initial position

# Get frame dimensions for centering calculations
# Note: This should be done after cam.read() for better robustness
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width // 2
center_y = frame_height // 2

# Initial Servo Angles (assuming 90 degrees is the center position)
current_pan = 90
current_tilt = 90
send_pan_tilt(current_pan, current_tilt) # Send initial position

# Toggle flags
show_red = False
show_face = False
show_eye = False
show_smile = False

# Tracking parameters (used for the serial commands)
TRACKING_SENSITIVITY = 0.5 # Smaller value means less movement per frame
MIN_AREA_FOR_TRACKING = 500 # Only track if the object is big enough

while True:
    ret, img = cam.read()
    if not ret:
        break

    # Convert to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- Tracking Target Initialization ---
    # Centroid of the target we want to track (if any)
    target_cx = None 
    target_cy = None 
    
    # ... (Your existing detection code follows) ...

    # Face Detection (Priority 1 for tracking)
    if show_face:
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        # We'll track the largest face found
        if len(faces) > 0:
            # Find the largest face (x, y, w, h)
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Draw rectangle and text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Face detected", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
            
            # Calculate centroid for tracking
            target_cx = x + w // 2
            target_cy = y + h // 2

    # Red Object Detection (Priority 2 for tracking, only if no face is tracked)
    elif show_red: 
        # ... (Your existing red object detection code) ...
        mask1 = cv2.inRange(imgHSV, lower_red1, upper_red1)
        mask2 = cv2.inRange(imgHSV, lower_red2, upper_red2)
        mask = mask1 + mask2

        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelo)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelc)

        contours, hierarchy = cv2.findContours(maskClose.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_contour = None
        max_area = MIN_AREA_FOR_TRACKING
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_contour = cnt

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(img, "Object detected", (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 1)

            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                # Calculate centroid for tracking
                target_cx = int(M["m10"] / M["m00"])
                target_cy = int(M["m01"] / M["m00"])
                
                # Draw centroid circle and text
                cv2.circle(img, (target_cx, target_cy), 7, (0, 0, 255), -1)
                cv2.putText(img, f"Centroid: ({target_cx},{target_cy})", (target_cx+30, target_cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        cv2.imshow("Mask", mask)
        cv2.imshow("Mask Open", maskOpen)
        cv2.imshow("Mask Close", maskClose)


    # Eye and Smile Detection (Just drawing, not for tracking)
    if show_eye:
        eyes = eye_cascade.detectMultiScale(img, 1.1, 5)
        for x, y, w, h in eyes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Eyes detected", (x-40, y-40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

    if show_smile:
        smiles = smile_cascade.detectMultiScale(img, 1.1, 5)
        for x, y, w, h in smiles:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Smile detected", (x-40, y-40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

    # --- Pan-Tilt Logic ---
    if target_cx is not None and target_cy is not None:
        # Calculate the error (difference between target and center)
        error_x = target_cx - center_x
        error_y = target_cy - center_y

        # Use the error to adjust the pan and tilt angles
        # A positive error_x (target is to the right) means Pan angle needs to increase (move right)
        # A positive error_y (target is below) means Tilt angle needs to decrease (move down)
        
        # Adjust Pan
        # We divide by frame_width/2 to normalize the error, then scale by sensitivity
        pan_correction = error_x / (frame_width / 2) * TRACKING_SENSITIVITY 
        current_pan += pan_correction
        
        # Adjust Tilt (Note the inversion: larger Y means moving down in the image, which is usually a smaller tilt angle)
        tilt_correction = error_y / (frame_height / 2) * TRACKING_SENSITIVITY 
        current_tilt -= tilt_correction 

        # Send the new angles to the Arduino
        send_pan_tilt(current_pan, current_tilt)
        
        # Draw the center point and a line to the target for visual feedback
        cv2.circle(img, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.line(img, (center_x, center_y), (target_cx, target_cy), (255, 0, 255), 2)

    cv2.imshow("Original", img)

    # ... (Your existing key handling code) ...
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
if arduino_serial is not None:
    arduino_serial.close()
    print("Serial port closed.")