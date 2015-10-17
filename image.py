import numpy as np
import cv2

# 0 - black
# 255 - white

THRESHOLD = 75

def rgb2gray(frame):
    return np.inner(frame, [0.2989, 0.587, 0.114]).astype(np.uint8)	

def binarize(gray):
    return np.where(gray < THRESHOLD, gray, 255)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    gray = rgb2gray(frame)
    gray_bin = binarize(gray)
    # t = otsu(gray_bin)
    
    # Display the resulting frame
    cv2.imshow('frame', gray_bin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
