import cv2
import numpy as np

for i in [0,4,5]:
    img = cv2.imread("label_{0}.png".format(i))
    img = cv2.resize(img, (128, 128))

    print(img.shape)

    img.tofile("label_{0}.bin".format(i))
