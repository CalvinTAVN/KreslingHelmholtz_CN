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
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

lower_green = np.array([45, 100, 100])
upper_green = np.array([75, 255, 255])


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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect both lines
        orange_dirs, orange_centers = detect_lines(hsv.copy(), lower_orange, upper_orange, (0, 165, 255))
        green_dirs, green_centers = detect_lines(hsv.copy(), lower_green, upper_green, (0, 255, 0))

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

            # Draw the center point
            cv2.circle(frame, tuple(center_point), 5, (255, 0, 255), -1)

            # Draw the direction vector
            length = 100  # length of the arrow
            tip = (center_point + avg_dir * length).astype(int)
            cv2.arrowedLine(frame, tuple(center_point), tuple(tip), (255, 0, 255), 3, tipLength=0.2)

        cv2.imshow("Vector Between Parallel Lines", frame)


        key = cv2.waitKey(1) 
        if (key == ord('s')):
                print("Breaking")
                break

finally:
    vid.release()   
    # Destroy all the windows 
    cv2.destroyAllWindows() 
    






# Show result
cv2.imshow("Vector Between Parallel Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
