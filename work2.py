import numpy as np
import cv2
from otsu import otsu

# 0 - black
# 255 - white

def rgb2gray(frame):
    return np.inner(frame, [0.2989, 0.587, 0.114]).astype(np.uint8)

def binarize(gray):
    threshold = otsu(gray)
    # threshold = 55
    return np.where(gray < threshold, np.uint8(255), np.uint8(0))

def inept_or_not(rect):
    w = rect[2]
    h = rect[3]

    return (10 <= w <= 500) and (10 <= h <= 500) and (0.5 <= float(w) / h <= 2.0)

def filter_inept_rects(rects):
    return list(filter(inept_or_not, rects))

# im = cv2.imread("real.jpg")
im = cv2.imread("photo_2.jpg")


im_gray = rgb2gray(im)
im_th = binarize(im_gray)

cv2.imshow("Thresholded image", im_th)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

print(rects)

rects = filter_inept_rects(rects)


for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2) 


cv2.imshow("Resulting Image with Rectangular ROIs", im)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
    
#     gray = rgb2gray(frame)
#     gray_bin = binarize(gray)

#     # edges = cv2.Canny(gray_bin,100,200)
#     # Display the resulting frame
#     cv2.imshow('frame', gray_bin)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
