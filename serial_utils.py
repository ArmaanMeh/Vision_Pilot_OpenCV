import serial
import time

class SerialComm:
    def __init__(self, port="COM3", baudrate=9600, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # wait for Arduino reset

    def send(self, tag, x, y):
        """
        Send formatted message to Arduino.
        Example: FACE,320,240
        """
        message = f"{tag},{x},{y}\n"
        self.ser.write(message.encode())

    def close(self):
        self.ser.close()
