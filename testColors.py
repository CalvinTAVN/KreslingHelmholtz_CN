import cv2
import numpy as np
import time

def tune_color_range(image, window_name="Tuner"):
    """
    Opens a trackbar UI to help find HSV lower/upper bounds interactively.
    Returns the chosen lower and upper HSV bounds.
    """
    def nothing(x):
        pass

    cv2.namedWindow(window_name)

    # Create 6 trackbars for H, S, V min and max
    cv2.createTrackbar("H_min", window_name, 0, 179, nothing)
    cv2.createTrackbar("H_max", window_name, 179, 179, nothing)
    cv2.createTrackbar("S_min", window_name, 0, 255, nothing)
    cv2.createTrackbar("S_max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V_min", window_name, 0, 255, nothing)
    cv2.createTrackbar("V_max", window_name, 255, 255, nothing)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    while True:
        # Get current positions
        h_min = cv2.getTrackbarPos("H_min", window_name)
        h_max = cv2.getTrackbarPos("H_max", window_name)
        s_min = cv2.getTrackbarPos("S_min", window_name)
        s_max = cv2.getTrackbarPos("S_max", window_name)
        v_min = cv2.getTrackbarPos("V_min", window_name)
        v_max = cv2.getTrackbarPos("V_max", window_name)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        cv2.imshow(window_name, result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"Lower HSV: {lower}")
            print(f"Upper HSV: {upper}")
            break

    cv2.destroyWindow(window_name)
    print(f"lower: {lower}   upper: {upper}")
    return lower, upper


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
        lower, upper = tune_color_range(frame, "orange")
finally: 
    vid.close()
    print("closing")