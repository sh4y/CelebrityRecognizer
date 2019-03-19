import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

def get_video_frames(file):
    capture = cv2.VideoCapture(file)
    result, image = capture.read()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    count = 0
    frames[count] = image
    while result:
        result, image = capture.read()
        count += 1
        frames[count] = image
    return frames

frames = get_video_frames('C:\Users\Shay\Documents\CSC420\Project\\test.mp4')
frame = frames[0]
plt.imshow(frame)
plt.show()
