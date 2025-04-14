import cv2
import numpy as np

cap = cv2.VideoCapture("your_video.mp4")

ret, prev = cap.read()
if not ret:
    raise ValueError("Couldn't read video")

# Convert first frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Create blank accumulator (same shape as input)
motion_accum = np.zeros_like(prev_gray, dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Accumulate motion
    motion_accum = cv2.bitwise_or(motion_accum, thresh)

    prev_gray = gray

cap.release()

# Optional: convert to color for display
motion_color = cv2.cvtColor(motion_accum, cv2.COLOR_GRAY2BGR)
cv2.imshow("Motion Summary", motion_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
