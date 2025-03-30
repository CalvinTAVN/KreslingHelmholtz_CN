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

     
 
print("Hello World \n")
print("OpenCV version: ", cv2.__version__ )

    
  
vid = cv2.VideoCapture(1) 
fps = vid.get(cv2.CAP_PROP_FPS)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
print(f"fps: {fps}")
print("width: ", width)
print("height: ", height)
    
                                                                                                                                                   




ret, frame = vid.read() 
sample_time = 0.02 #sample time is 100 ms
prev_time = time.time()
current_time = time.time()


while(True):
    while(current_time-prev_time < sample_time):
        current_time = time.time()
    prev_time = current_time    	        
    # Capture the video frame by frame
    ret, frame = vid.read() 
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    org2 = (740, 280)
    
    x_xaxis = 280
    y_xaxis = 900
    
    x_yaxis = x_xaxis
    y_yaxis = y_xaxis - 68
    
    fontScale = 1
    color = (255, 0, 0)
    #color2 = (0, 0, 255)
    thickness = 3
    
    
    
    
    start_point = (x_xaxis, y_xaxis)  
    end_point_x = (x_xaxis+50, y_xaxis)
    end_point_y = (x_yaxis, y_xaxis + 50)    
  
    image = cv2.arrowedLine(frame, start_point, end_point_x, color, thickness)  
    image = cv2.arrowedLine(frame, start_point, end_point_y, color, thickness)  
    
    
    
    
    image = cv2.putText(frame, 'x', (x_xaxis + 60, y_xaxis+5), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    image = cv2.putText(frame, 'y', (x_xaxis + 12, y_xaxis+55), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
   
        
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) & 0xFF == ord('s'): 
            break
    


# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 






































