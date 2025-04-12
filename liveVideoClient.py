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
import select

def get_most_recent_line(sock_file, timeout=0.01):
    latest_line = None

    # Check if data is available within timeout
    rlist, _, _ = select.select([sock_file], [], [], timeout)
    if not rlist:
        return None  # No data ready

    # Drain the available buffer
    while True:
        rlist, _, _ = select.select([sock_file], [], [], 0)
        if not rlist:
            break  # No more immediately available data

        line = sock_file.readline()
        if not line:
            break  # EOF or connection closed

        latest_line = line  # keep only the latest

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
        latest_line = get_most_recent_line(sock_file)
        if not latest_line:
            vec_unit = [0, 0]
        else:
            vec_unit = np.array(latest_line)
        true_vec_unit = np.array([vec_unit[1], -vec_unit[0]])
        print("Latest vec_unit:", vec_unit)
        print("true_vec_unit: ", true_vec_unit)
        motion = input("s to stop, c to compress, u to uncompress:")
        if (motion == 's'):
            true_break = True
            break
                
        elif (motion == 'c'):
            uncompressedRotationVec = detect.rotate_vector_counterclockwise(true_vec_unit, 91.0685)
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
            print("true vec: ", true_vec_unit)
            print('uncompressed: ', uncompressedRotationVec)
            [x1, x2, y1, y2, z1, z2] = encode.con([x, y, z], n)
            encode.sendCAN(x1, y1, z1, can = can, bus = bus)
        elif (motion == 'u'):
            uncompressedRotationVec = detect.rotate_vector_clockwise(true_vec_unit, 110)
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
            print("true vec: ", true_vec_unit)
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

