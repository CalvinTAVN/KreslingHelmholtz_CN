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

#create a socket
import socket
import json

#set up camera
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

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('localhost', 9998))      # IP address and port
server.listen(1)
print("Waiting for connection...")

conn, addr = server.accept()
print("Connected by", addr)
json_string = json.dumps([0, 0, 0])
conn.sendall((json_string + '\n').encode('utf-8'))

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

try:
    while True:
        while(current_time-prev_time < sample_time):
            current_time = time.time()
        prev_time = current_time    	        
        # Capture the video frame by frame
        ret, frame = vid.read() 
        #note vec unit of image is +y is right, +y is down
        processed_frame, vec_unit = detect.process_videoAruco2(frame, mtx, dist, detector)
        vec_unit = np.ndarray([float(motion), vec_unit[0], vec_unit[1]])
        #note frame of actual Helmholtz Coil is +x is down, + y is left
        json_string = json.dumps(vec_unit.tolist())
        conn.sendall((json_string + '\n').encode('utf-8'))

        cv2.imshow('frame', processed_frame)
        key = cv2.waitKey(1) 
        if (key == ord('s')):
                print("Breaking")
                break

        elif (key == ord('p')):
            recVideo = True
            print("Recording Video Now")
finally: 
    print("closing, turning everything off")
    conn.close()
    server.close()
    vid.release()   
    # Destroy all the windows 
    cv2.destroyAllWindows() 


    

