#stuff needed for detection Functions
import os
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import sys
#import cv2.aruco as aruco
import matplotlib.pyplot as plt
import detectionFunctions as detect
import time 

#stuff needed for encode Functions
import spidev
import pandas as pd
import can
from scipy.spatial.transform import Rotation as R
import encodeFunctions as encode

#to create a socket
import socket
import json

def get_latest_message(sock_file):
    latest_line = None
    while True:
        line = sock_file.readline()
        if not line:
            break
        latest_line = line

    if latest_line:
        return json.loads(latest_line)
    else:
        return None


#starting can bus
bus = can.interface.Bus(bustype='socketcan', channel='can1', bitrate=1000000)
bus.shutdown()
bus = can.interface.Bus(bustype='socketcan', channel='can1', bitrate=1000000)
print("Canbus Successfully Setup. \n")

#initial Values of helmholtz coils
values = [0, 0, 0, 0, 0, 0]
tx = encode.encodeNum(values)
message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
bus.send(message, timeout = 0.5)
time.sleep(0.01)
print("initial values are now all 0s")

#setting up socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9998))  # Connect to server
sock_file = client.makefile(mode='r')

true_break = False
try:
    while True:
        latest_line = get_latest_message(sock_file, client)
        if not latest_line:
            vec_unit = [0, 0]
        else:
            parsed = json.loads(latest_line)
            vec_unit = np.array(parsed)
        print("Latest vec_unit:", vec_unit)
        motion = input("Enter 'r' for rolling, 't' for spinning, 'c' for constant field,  or 's' to stop:")
        if (motion == 's'):
            true_break = True
            break
                
        elif (motion == 'c'):
            uncompressedRotationVec = detect.rotate_vector_clockwise(vec_unit, 91.0685)
            x = uncompressedRotationVec[0]
            y = uncompressedRotationVec[1]
            z = 0

            a = input("amplitude: ")
            a = int(a)
            n = input("Number of samples:")
            n = int(n)

            x = a * x
            y = a * y
            print("original vec: ", vec_unit)
            print('uncompressed: ', uncompressedRotationVec)
            [x1, x2, y1, y2, z1, z2] = encode.con([x, y, z], n)
            encode.sendCAN(x1, y1, z1, can = can, bus = bus)
        elif (motion == 'u'):
            uncompressedRotationVec = detect.rotate_vector_counterclockwise(vec_unit, 118.9315)
            x = uncompressedRotationVec[0]
            y = uncompressedRotationVec[1]
            z = 0

            a = input("amplitude: ")
            a = int(a)
            n = input("Number of samples:")
            n = int(n)

            x = a * x
            y = a * y
            print("original vec: ", vec_unit)
            print('compressed: ', uncompressedRotationVec)
            [x1, x2, y1, y2, z1, z2] = encode.con([x, y, z], n)
            encode.sendCAN(x1, y1, z1, can = can, bus = bus)
        if true_break:
            break
finally:
    print("breaking everything")
    bus.shutdown()
    sock_file.close()
    client.close()

