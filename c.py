import numpy as np
import cv2
from otsu import otsu
import math
import pickle
from network import *

from ctypes import *
from numpy.ctypeslib import ndpointer

cdll.LoadLibrary('WorkingNetwork.so')

libc = CDLL('WorkingNetwork.so')

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

    return (15 <= w <= 500) and (15 <= h <= 500) and (0.5 <= float(w) / h <= 2.0)

def filter_inept_rects(rects):
    return list(filter(inept_or_not, rects))

def draw_rectangle(im, rect):
    c = (0, 0, 255)
    y, x = rect[0], rect[1]
    w, h = rect[2], rect[3]

    for i in range(x, x + h):
        for j in range(-1, 2):
            try:
                im[i, y + j] = c
            except:
                pass
            try:
                im[i, y + w + j] = c
            except:
            	pass

    for i in range(y, y + w):
        for j in range(-1, 2):
            im[x + j, i] = c
            im[x + h + j, i] = c

def rotation_matrix(img, angle):
    center_x, center_y = img.shape[0] / 2.0, img.shape[1] / 2.0
    alpha = math.cos(math.radians(angle))
    betta = math.sin(math.radians(angle))

    return np.array([[alpha, betta, (1 - alpha)*center_x - betta*center_y],
                     [-betta, alpha, betta*center_x + (1 - alpha)*center_y]])

def rotate_im(img, angle):
    rows, cols = img.shape
    M = rotation_matrix(img, angle)
    r = cv2.warpAffine(img, M, (cols, rows))

    return r


def network_output(pyarr):
    arr = (c_double * len(pyarr))(*pyarr)
    libc.neuralNetwork.restype = ndpointer(dtype=c_double, shape=(10,))
    return libc.neuralNetwork(arr)


def recognize_digit(img):

    return np.argmax(network_output(img))



im = cv2.imread("photo_2.jpg")


im_gray = rgb2gray(im)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 1)
im_th = binarize(im_gray)


cv2.imshow("Thresholded image", im_gray)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.imshow("Thresholded image", im_th)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# print(rects)

rects = filter_inept_rects(rects)

for rect in rects:
    draw_rectangle(im, rect)

# black canvas to draw all found ROIs
black_img = np.array([[np.uint8(255)]*1200]*500)

i = 0
j = 0
for rect in rects:
    y, x, w, h = rect[:]
    w2, h2 = int(w * 0.35), int(h * 0.35)

    r = im_th[x-h2:x+h+h2, y-w2:y+w+w2]
    r = cv2.resize(r, (28, 28), interpolation=cv2.INTER_CUBIC)

    black_img[j:j+28, i:i+28] = r
    i += 50
    if i >= 900:
        i = 0
        j += 50


    # converting pixels into (0.0, 1.0) range
    roi = r.reshape((784, 1))/255.0
    predicted_digit = recognize_digit(roi)

    cv2.putText(im, str(predicted_digit), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)


cv2.imshow("Resulting Image with Rectangular ROIs", black_img)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.imshow("Resulting Image with Rectangular ROIs", im)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
