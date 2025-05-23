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


print("OpenCV version:", cv2.__version__)

#calibration from new live Calibration
mtx = [[605.61062509, 0.00000000e+00, 331.38868735],
 [0.00000000e+00, 601.78883325, 244.80436645],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
mtx = np.array(mtx)

dist = [-0.33923236,  0.07590603, -0.00238233, -0.00349838,  0.19293399]
dist = np.array(dist)

#initialize Detector
cv2.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

parameters = aruco.DetectorParameters()
#parameters.adaptiveThreshWinSizeMin = 10
#parameters.adaptiveThreshWinSizeMax = 23
#parameters.adaptiveThreshWinSizeStep = 10
#parameters.minMarkerPerimeterRate = 0.05
#parameters.maxMarkerPerimeterRate = 0.2
#parameters.polygonalApproxAccuracyRate = 0.1
#parameters.minCornerDistanceRate = 0.05
#parameters.minMarkerDistanceRate = 0.5
#parameters.minDistanceToBorder = 3
#parameters.markerBorderBits = 1
#parameters.minOtsuStdDev = 5.0
#parameters.perspectiveRemoveIgnoredMarginPerCell = 0.2
detector = aruco.ArucoDetector(cv2.aruco_dict, parameters)

#display video with Arucomarkers
motion = input("input camera Number(0 side View, 1 top View): ")
while True:
    try: 
        motion = int(motion)
        break
    except:
        print("Error: Invalid Input")
        print("setting default value to 0")
        motion = 0

vid = cv2.VideoCapture(motion) 
fps = vid.get(cv2.CAP_PROP_FPS)
print(f"fps: {fps}")
                                                                                                                                          

video = []
ret, frame = vid.read() 
sample_time = 0.1 #sample time is 100 ms
prev_time = time.time()
current_time = time.time()

recVideo = False

#start Bus
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

try:
    while(True):
        while(current_time-prev_time < sample_time):
            current_time = time.time()
        prev_time = current_time    	        
        # Capture the video frame by frame
        ret, frame = vid.read() 
        #note vec unit of image is +y is right, +y is down
        processed_frame, vec_unit = detect.process_videoAruco2(frame, mtx, dist, detector)

        #note frame of actual Helmholtz Coil is +x is down, +y is left
        true_vec_unit = np.array([vec_unit[1], -vec_unit[0]])
        #note (-1, -1) of image points straight up on image, meaning +y is down and +x is right
        

        cv2.imshow('frame', processed_frame) 

        key = cv2.waitKey(1)

        if (key == ord('s')):
            print("Breaking")
            break

        elif (key == ord('p')):
            recVideo = True
            print("Recording Video Now")
        #if key is t, compress the object
        elif (key == ord('c')):
            #since on the magnet, the x is flipped, we intead rotate CCW
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
            print("true vec: ", true_vec_unit)
            print('compressed: ', uncompressedRotationVec)
            [x1, x2, y1, y2, z1, z2] = encode.con([x, y, z], n)
            encode.sendCAN(x1, y1, z1, can = can, bus = bus)
        elif (key == ord('u')):
            uncompressedRotationVec = detect.rotate_vector_counterclockwise(true_vec_unit, 91.0685)
except:
    print("bus went wrong, shutting down")


bus.shutdown()

vid.release()   
# Destroy all the windows 
cv2.destroyAllWindows() 

    