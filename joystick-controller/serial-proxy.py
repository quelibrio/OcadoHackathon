import serial
import time
from sys import stdout
import datetime;

try:
        ser = serial.Serial('/dev/ttyACM1', 115200, timeout = 1)
except:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout = 1)

while True:
        try:
                readOut = ser.readline().decode('ascii')
                print(readOut)
                stdout.flush()
        except:
                pass
