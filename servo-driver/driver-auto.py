from sys import stdin
from serial import Serial
import time
from sys import stdout
import datetime;
x0 = 506
y0 = 526
try:
    ser = Serial("/dev/ttyUSB0", 9600)
except:
    ser = Serial("/dev/ttyUSB1", 9600)

width = 910
height = 512
while True:
    try:
        readOut = stdin.readline()
        if "Downloading" in readOut:
            continue
        try:
            x1,y1,x2,y2=readOut.strip().split(" ")
        except:
            continue
        #width 910
        #height 512
        print(x1,y1,x2,y2)
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        #horizontal centring
        centerOffset = (x2 + x1) / 2 - (width / 2)
        cmd = 'a'
        if (centerOffset > width / 4):
            cmd = 'd'
        if (centerOffset < -width / 4):
            cmd = 'e'
        if (x2 < height * 0.8 and x1 < height * 0.2):
            cmd = 'b'
        print(cmd)
        ser.write(dir)
    except Exception as e:
        print(str(e))
        pass
