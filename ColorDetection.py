import cv2
import numpy as np
import time

def detect_lines(image, lower_hsv, upper_hsv, bgr_color):
    """Detects lines of a given color, returns list of line vectors and centers"""
    mask = cv2.inRange(image, lower_hsv, upper_hsv)
    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    directions = []
    centers = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            directions.append(np.array([dx, dy]))
            centers.append(np.array([(x1 + x2) / 2, (y1 + y2) / 2]))
            cv2.line(image, (x1, y1), (x2, y2), bgr_color, 2)
    
    return directions, centers

# HSV ranges (tweak as needed)
lower_orange = np.array([0, 11, 247])
upper_orange = np.array([22, 187, 255])

lower_green = np.array([75, 6, 242])
upper_green = np.array([94, 255, 255])


#buffer
center_history = []
dir_history = []
buffer_size = 5  

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
true_break = False
try:
    while True:
        while(current_time-prev_time < sample_time):
            current_time = time.time()
        prev_time = current_time    	        
        # Capture the video frame by frame
        ret, frame = vid.read() 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #gaussian blur the image
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # Detect both lines
        orange_dirs, orange_centers = detect_lines(blurred.copy(), lower_orange, upper_orange, (0, 165, 255))
        green_dirs, green_centers = detect_lines(blurred.copy(), lower_green, upper_green, (0, 255, 0))

        if orange_dirs and green_dirs:
            # Average direction vector (normalize)
            all_dirs = orange_dirs + green_dirs
            avg_dir = np.mean(all_dirs, axis=0)
            norm = np.linalg.norm(avg_dir)
            if norm != 0:
                avg_dir = avg_dir / norm

            # Find midline between both color groups
            all_centers = orange_centers + green_centers
            center_point = np.mean(all_centers, axis=0).astype(int)

            center_history.append(center_point)
            dir_history.append(avg_dir)

            # Keep history length fixed
            if len(center_history) > buffer_size:
                center_history.pop(0)
                dir_history.pop(0)

            smoothed_center = np.mean(center_history, axis=0).astype(int)
            smoothed_dir = np.mean(dir_history, axis=0)
            smoothed_dir = smoothed_dir / np.linalg.norm(smoothed_dir) if np.linalg.norm(smoothed_dir) != 0 else smoothed_dir

            # Draw the center point
            cv2.circle(frame, tuple(smoothed_center), 5, (255, 0, 255), -1)

            # Draw the direction vector
            length = 100  # length of the arrow
            tip = (smoothed_center + avg_dir * length).astype(int)
            cv2.arrowedLine(frame, tuple(smoothed_center), tuple(tip), (255, 0, 255), 3, tipLength=0.2)

        cv2.imshow("Vector Between Parallel Lines", frame)


        key = cv2.waitKey(1) 
        if (key == ord('s')):
            print("Breaking")
            true_break = True
            break
        if true_break:
            break

finally:
    vid.release()   
    # Destroy all the windows 
    cv2.destroyAllWindows() 
    






# Show result
cv2.imshow("Vector Between Parallel Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
