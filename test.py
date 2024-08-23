import torch
import psutil
import os
import cv2

path = "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-13_22_36/images/Azure_0_orig/0.png"

img = cv2.imread(path)[:, 100:370]
cv2.imshow("1", img)
cv2.waitKey(0)
