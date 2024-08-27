# import torch
# import psutil
# import os
# import cv2

# path = "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-13_22_36/images/Azure_0_orig/0.png"

# img = cv2.imread(path)[:, 100:370]
# cv2.imshow("1", img)
# cv2.waitKey(0)

# import psutil

# process = psutil.Process()
# print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


# import torch

# t = torch.cuda.get_device_properties(0).total_memory
# r = torch.cuda.memory_reserved(0)
# a = torch.cuda.memory_allocated(0)
# f = r - a  # free inside reserved
# print(str(t / 1024**2) + "\n")
# print(str(r) + "\n")
# print(str(a) + "\n")


print((1, 2, 3, 4)[:-1])
