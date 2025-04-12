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

#starting can bus
bus = can.interface.Bus(bustype='socketcan', channel='can1', bitrate=1000000)
bus.shutdown()
bus = can.interface.Bus(bustype='socketcan', channel='can1', bitrate=1000000)
print("Canbus Successfully Setup. \n")

#initial Values of helmholtz coils
values = [0, 0, 0, 0, 0, 0]
tx = encode.encode(values)
message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
bus.send(message, timeout = 0.5)
time.sleep(0.01)
print("initial values are now all 0s")

#setting up socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))  # Connect to server


try:
    while True:
        received = client.recv(1024)
        parsed = json.loads(received.decode('utf-8'))
        print(parsed)
        motion = input("Enter 'r' for rolling, 't' for spinning, 'c' for constant field,  or 's' to stop:")
        if (motion == 's'):
            break
except:
    print("breaking everything")

bus.shutdown()
client.close()