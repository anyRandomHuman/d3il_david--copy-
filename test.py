from real_robot_env.robot.hardware_azure import Azure
import cv2

cam0 = Azure(device_id=2)
cam0.connect()
img = cam0.get_sensors()["rgb"][:, :, :3]
cv2.imwrite("test.jpg", img)
