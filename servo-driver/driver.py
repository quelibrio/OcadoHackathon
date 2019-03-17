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

while True:
    try:
        readOut = stdin.readline()
        if not ',' in readOut:
            continue
        click,x,y=readOut.strip().split(",")

        print(click,x,y)
        x = int(x)
        y = int(y)
        dy = y - y0
        dx = x - x0
        #print("left: ", lm, "right: ", rm)
        #ser.write(buf)
        abs(0)
        horizontal = abs(dx) > abs (dy)
        vertical = abs(dy) > abs (dx)
        if (horizontal and dx > 10):
            dir = b'd'
        elif (horizontal and dx < -10):
            dir = b'e'
        elif (vertical and dy > 10):
            dir = b'b'
        elif (vertical and dy < -10):
            dir = b'c'
        else:
            dir = b'a'
        ser.write(dir)
        print(dir)
    except:
        pass
