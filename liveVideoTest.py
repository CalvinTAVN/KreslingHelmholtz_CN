import spidev
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import can
import sys
import os
import math

import cv2
import cv2.aruco as aruco


#calibration from new live Calibration
mtx = [[605.61062509, 0.00000000e+00, 331.38868735],
 [0.00000000e+00, 601.78883325, 244.80436645],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
mtx = np.array(mtx)

dist = [-0.33923236,  0.07590603, -0.00238233, -0.00349838,  0.19293399]
dist = np.array(dist)
print("Hello World \n")
print("OpenCV version: ", cv2.__version__ )

def undistortImage(frame, matrix_coefficients, distortion_coefficients):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix_coefficients, distortion_coefficients, (w,h), 1, (w,h))
    dst = cv2.undistort(frame, matrix_coefficients, distortion_coefficients, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst     
  
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
    #unDistort Image does not work right now
    #new_frame = undistortImage(frame, mtx, dist)

    
        
    cv2.imshow('frame', frame)
    if recVideo == True: 
        video.append(frame)
    key = cv2.waitKey(1)

    if (key == ord('s')):
        print("Breaking")
        break

    elif (key == ord('p')):
        recVideo = True
        print("Recording Video Now")
    
    


# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

if recVideo:
    fileName = input("give file name: ")
    videoOutputFile = '/home/kostas/Documents/KreslingHelmholtz_CN/Videos/3_30_videos' + fileName + '.avi'
    videoLength = len(video)
    width = len(video[0])
    length = len(video[0][0])
    print(f"vidLength: {videoLength} width: {width} length: {length}")
    print(f"single frame type{video[0].shape}")
    out = cv2.VideoWriter(videoOutputFile,  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         1/sample_time, (length, width)) 
    for frame in video:
        out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()







































