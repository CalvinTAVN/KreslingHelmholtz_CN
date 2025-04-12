import os
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import sys
#import cv2.aruco as aruco
import matplotlib.pyplot as plt
import detectionFunctions.py

print("OpenCV version:", cv2.__version__)

#calibration from new live Calibration
mtx = [[605.61062509, 0.00000000e+00, 331.38868735],
 [0.00000000e+00, 601.78883325, 244.80436645],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
mtx = np.array(mtx)

dist = [-0.33923236,  0.07590603, -0.00238233, -0.00349838,  0.19293399]






