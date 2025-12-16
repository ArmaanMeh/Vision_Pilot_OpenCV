# 🤖 Real-Time Vision-Based Pan/Tilt Tracker (OpenCV + Arduino)

This Python script is designed to run on a host PC, using **OpenCV** for real-time video processing and object/face detection. It then uses **PySerial** to send computed pan and tilt position commands over a serial port to a connected microcontroller (like an Arduino) controlling a camera mount with two servos.

The control mechanism utilizes a highly smoothed Proportional-Integral (PI) architecture to achieve precise, non-jerky object tracking.

## 🧰 Prerequisites and Setup

1.  **Python 3.x**
2.  **Libraries:** Install required Python packages: `pip install opencv-python numpy pyserial`
3.  **Haar Cascades:** Ensure the required XML files (`haarcascade_frontalface_default.xml`, etc.) are located in the `Haar_Cascades XML/` folder relative to the script.
4.  **Hardware:** A pan/tilt servo mount connected to an Arduino or similar microcontroller, running a sketch capable of reading the absolute position command over serial.

## ⚙️ Configuration Block

This block defines all the configurable constants for hardware communication, control tuning, and vision thresholds.

| Category | Constant | Description |
| :--- | :--- | :--- |
| **Hardware** | `SERIAL_PORT` | The COM port for the microcontroller (e.g., `'COM8'`). |
| | `BAUD_RATE` | Communication speed (must match the device). |
| | `CAMERA_ID` | Index of the webcam (usually `0` or `1`). |
| **Movement** | `SEND_RATE_HZ` | Position update frequency (e.g., 10 messages/sec). |
| | `DEAD_BAND_PIXELS` | Error threshold around the center to prevent jittering. |
| | `P_GAIN` | Proportional gain ($P$) for control aggressiveness. |
| | `MAX_SPEED_CAP` | Limits the maximum rate of change (velocity) to ensure smooth motion. |
| **Detection** | `LOWER_REDx`/`UPPER_REDx` | HSV ranges for red color segmentation. |
| | `MIN_AREA`/`MAX_AREA` | Filters valid contour sizes. |

```python
import cv2
import numpy as np
import serial
import time
import os 

# --- Hardware Settings ---
SERIAL_PORT = 'COM8' 	 	
BAUD_RATE = 9600
CAMERA_ID = 1

# --- Movement Settings ---
SEND_RATE_HZ = 10 	 	
SEND_INTERVAL = 1.0 / SEND_RATE_HZ 

DEAD_BAND_PIXELS = 35 
SMOOTHING_FACTOR = 0.5
MAX_SPEED_CAP = 0.22 	
P_GAIN = 0.0002 
delta_pan = 0.0
delta_tilt = 0.0

# --- Detection Settings ---
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

# Global state variables for toggles
toggle_red_ball = False
toggle_face = False
toggle_eyes = False
toggle_smile = False

# Global variables for smoothing
ser = None
last_send_time = 0
global_pan_position = 0.0
global_tilt_position = 0.0