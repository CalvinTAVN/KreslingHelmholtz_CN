import os
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import sys
#import cv2.aruco as aruco
import matplotlib.pyplot as plt


def undistortImage(frame, matrix_coefficients, distortion_coefficients):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix_coefficients, distortion_coefficients, (w,h), 1, (w,h))
    dst = cv2.undistort(frame, matrix_coefficients, distortion_coefficients, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst     

def estimatePoseSingleMarkers(corners, marker_size, mtx, dist):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        trashV, rot, tran = cv2.solvePnP(marker_points, c, mtx, dist, False, cv2.SOLVEPNP_IPPE_SQUARE)
        #print(f"trashV: {trashV}")
        #print(f"rot: {rot}")
        #print(f"tran: {tran}")
        rvecs.append(rot)
        tvecs.append(tran)
        trash.append(trashV)
        #return np.array([rvecs]), np.array([tvecs]), trash
        #convert items back into a single list
        rvecs = [item for sublist in rvecs for item in sublist]
        tvecs = [item for sublist in tvecs for item in sublist]
        rvecs = [item for sublist in rvecs for item in sublist]
        tvecs = [item for sublist in tvecs for item in sublist]
        return np.array(rvecs), np.array(tvecs), trash

#assuming image has already been undistorted
def pose_estimation(frame, aruco_dict_type, mtx, dist):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)

    parameters = aruco.DetectorParameters()

    #minimum window size for adaptive thresholding before finding contours (default 3).
    parameters.adaptiveThreshWinSizeMin = 10

    #maximum window size for adaptive thresholding before finding contours (default 23).
    parameters.adaptiveThreshWinSizeMax = 23

    #increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding (default 10).
    parameters.adaptiveThreshWinSizeStep = 10

    #determine minimum perimeter for marker contour to be detected. default 0.03
    #normal 0.05
    #0.085 if you only want to detect when cube is flat.
    parameters.minMarkerPerimeterRate = 0.05

    #determine maximum perimeter for marker contour to be detected. default 4.0
    parameters.maxMarkerPerimeterRate = 0.2

    #minimum accuracy during the polygonal approximation process to determine which contours are squares. (default 0.03)
    parameters.polygonalApproxAccuracyRate = 0.1

    #minimum distance between corners for detected markers relative to its perimeter (default 0.05)
    parameters.minCornerDistanceRate = 0.05

    #minimum average distance between the corners of the two markers to be grouped (default 0.125).
    #0.5 if you want to only see 1 marker on the cube, choose 0.125 otherwise
    parameters.minMarkerDistanceRate = 0.5

    #minimum distance of any corner to the image border for detected markers (in pixels) (default 3)
    #doesn't matter, markers are not outside image
    parameters.minDistanceToBorder = 3

    #number of bits of the marker border, i.e. marker border width (default 1). 
    #don't mess with
    parameters.markerBorderBits = 1
    
    #minimun standard deviation in pixels values during the decodification step to apply Otsu thresholding 
    # (otherwise, all the bits are set to 0 or 1 depending on mean higher than 128 or not) (default 5.0)
    parameters.minOtsuStdDev = 5.0

    #width of the margin of pixels on each cell not considered for the determination of the cell bit.
    #Represents the rate respect to the total size of the cell, i.e. perspectiveRemovePixelPerCell (default 0.13)
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.2

    detector = aruco.ArucoDetector(cv2.aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    #if markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            #estimate position for each marker and return values rvec and tvec
            rvec, tvec, trash = estimatePoseSingleMarkers(corners[i], 5, mtx, dist)
            aruco.drawDetectedMarkers(frame, corners, ids)
            #print(f"rvec: {rvec}")
            #print(f"tvec: {tvec}")
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 10)
    return frame

def pose_estimation2(frame, mtx, dist, aruco_detector):
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    vec_unit = np.array([0, 0])
    if ids is not None and len(corners) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            corner = corners[i][0]  # shape: (4, 2)
            id_number = int(ids[i][0])

            # Compute center of the marker
            center = corner.mean(axis=0)
            center_int = center.astype(int)

            # Vector along the top edge (top-left to top-right)
            vec_side = corner[1] - corner[0]
            vec_unit = vec_side / np.linalg.norm(vec_side)

            # Rotate 90° right (clockwise) if ID == 1
            if id_number == 1:
                vec_unit = np.array([vec_unit[1], -vec_unit[0]])

            # Compute end point of arrow for visualization
            end_point = (center + 50 * vec_unit).astype(int)

            # Draw marker ID
            cv2.putText(frame, str(id_number), (center_int[0], center_int[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw arrow
            cv2.arrowedLine(frame, tuple(center_int), tuple(end_point),
                            (255, 0, 0), 2, tipLength=0.3)

            # Draw center
            cv2.circle(frame, tuple(center_int), 4, (0, 255, 255), -1)

    return frame, vec_unit

def rotate_vector_clockwise(vec, angle_deg):
    """
    Rotates a 2D vector clockwise by a specified angle in degrees.

    Parameters:
    - vec: np.ndarray or list-like with 2 elements (e.g., [vx, vy])
    - angle_deg: float, angle in degrees to rotate clockwise

    Returns:
    - np.ndarray: rotated vector
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    vx, vy = vec
    rotated = np.array([
        vx * cos_theta + vy * sin_theta,
        -vx * sin_theta + vy * cos_theta
    ])
    return rotated

def rotate_vector_counterclockwise(vec, angle_deg):
    """
    Rotates a 2D vector counterclockwise by a specified angle in degrees.

    Parameters:
    - vec: np.ndarray or list-like with 2 elements [vx, vy]
    - angle_deg: float, angle in degrees to rotate counterclockwise

    Returns:
    - np.ndarray: rotated vector
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    vx, vy = vec
    rotated = np.array([
        vx * cos_theta - vy * sin_theta,
        vx * sin_theta + vy * cos_theta
    ])
    return rotated


#testing on a video
def process_video(input_video_path, output_video_path, mtx, dist):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open Video.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / frame_rate

    print("frames = ", frame_count)    
    print("fps = ", frame_rate)
    print("duration = ", duration)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    create_out = True
    #out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("end of Video")
            break

        counter = counter + 1
        
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        undistortedGray = undistortImage(frame, mtx, dist)
        
        if create_out == True:
            width = undistortedGray[0,:,0].shape[0] 
            height = undistortedGray[:,0,0].shape[0]
            print(f"width: {width}")
            print(f"height: {height}")
            out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
            create_out = False
        #print(f"frameshape: {frame.shape}")
        #print(f"undistorted Shape: {undistortedGray.shape}")
        frame = pose_estimation(undistortedGray, aruco.DICT_4X4_50, mtx, dist)
        out.write(frame)
    cap.release()
    out.release()
    print(counter)
    return(counter)

#calculates the boundingbox to find the arucoMarkers in
def calculateBoundingBox(corners, id, dict, frame_width, frame_height):
    #print(f"corner: {corner}")
    point1 = corners[0][0]
    point2 = corners[0][2]
    #print(f"point1: {point1}")
    #print(f"point2: {point2}")
    center = (point1 + point2) / 2
    center = [int(center[0]), int(center[1])]
    length = 2 * int(math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2))
    #calculate box
    topLeft = [max(0, min(center[0] - length, frame_width - 1)), max(0, min(center[1] - length, frame_height - 1))]
    topRight = [max(0, min(center[0] + length, frame_width - 1)), max(0, min(center[1] - length, frame_height - 1))]
    botRight = [max(0, min(center[0] + length, frame_width - 1)), max(0, min(center[1] + length, frame_height - 1))]
    botLeft = [max(0, min(center[0] - length, frame_width - 1)), max(0, min(center[1] + length, frame_height - 1))]
    #print(f"id: {ids[i]}")
    #print(f"topLeft: {topLeft} topRight: {topRight} botRight: {botRight} botLeft: {botLeft}")
    newCorner = np.array([[ topLeft, topRight, botRight, botLeft ]])
    # - - , + -, + +, - +
    dict[id[0]] = center, newCorner

    return dict
    




def process_videoAruco(input_image, mtx, dist):

    #undistortedGray = undistortImage(input_image, mtx, dist)
    frame = pose_estimation(input_image, mtx, dist)
    return frame

def process_videoAruco2(input_image, mtx, dist, detector):

    #undistortedGray = undistortImage(input_image, mtx, dist)
    frame, vec_unit = pose_estimation2(input_image, mtx, dist, detector)
    return frame, vec_unit

def process_video2(input_video_path, output_video_path, mtx, dist):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open Video.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / frame_rate

    print("frames = ", frame_count)    
    print("fps = ", frame_rate)
    print("duration = ", duration)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


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

    #initial compilation
    ret, frame = cap.read()
    if not ret:
        print("end of Video")
        return 0
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistortedGray = undistortImage(frame, mtx, dist)
    width = undistortedGray[0,:,0].shape[0] 
    height = undistortedGray[:,0,0].shape[0]
    print(f"width: {width}")
    print(f"height: {height}")
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    boundaryDict = {}
    boundingCorners, boundingIds = pose_estimation2(undistortedGray, mtx, dist, detector)
    print(f"boundingIdLength: {len(boundingIds)}")
    for i in range(len(boundingIds)):
        boundaryDict = calculateBoundingBox(boundingCorners[i], boundingIds[i], boundaryDict, width, height)
    out.write(frame)

    #counter will count every frame
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("end of Video")
            break

        counter = counter + 1
        
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        undistortedGray = undistortImage(frame, mtx, dist)

        #search all of the boundaries found in the dictionary
        for key, (center, newCorner) in boundaryDict.items():
            imageSnippet = undistortedGray[newCorner[0][0][1]:newCorner[0][2][1], newCorner[0][0][0]:newCorner[0][1][0]]
            newnewCorners, newnewIds = pose_estimation2(imageSnippet, mtx, dist, detector)
            #if not newnewCorners:
                #print("No new Corners Detected")
            #else:
                #print("Detected boundingbox Corners")   
            undistortedGray[newCorner[0][0][1]:newCorner[0][2][1], newCorner[0][0][0]:newCorner[0][1][0]] = imageSnippet
            
            #print(newCorner)
            #fordebug purposes
            outwardBox = np.expand_dims(newCorner, axis = 0)
            cv2.aruco.drawDetectedMarkers(undistortedGray, outwardBox)
        out.write(undistortedGray)

    cap.release()
    out.release()
    print(counter)
    return(counter)

    
