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

while(True):
    while(current_time-prev_time < sample_time):
         current_time = time.time()
    prev_time = current_time    	        
    # Capture the video frame by frame
    ret, frame = vid.read() 
    processed_frame, vec_unit = detect.process_videoAruco2(frame, mtx, dist, detector)
    print("imageFrame: ", vec_unit)
    true_vec_unit = np.array([vec_unit[1], -vec_unit[0]])
    print("trueFrame: ", true_vec_unit)
    #note (-1, -1) of image points straight up on image, meaning +y is down and +x is right
    cv2.imshow('frame', processed_frame) 

    key = cv2.waitKey(1)

    if (key == ord('s')):
        print("Breaking")
        break

    elif (key == ord('p')):
        recVideo = True
        print("Recording Video Now")


vid.release()   
# Destroy all the windows 
cv2.destroyAllWindows() 

    