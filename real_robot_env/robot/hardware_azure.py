import cv2
import pykinect_azure as pykinect
import logging
from typing import Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Azure():
    def __init__(self, device_id: str, name: Optional[str] = None):
        self.device_id = device_id
        self.name = name if name else f"cam{device_id}"

        self.__set_device_configuration() # sets self.device_config
        self.device = None
        

    def connect(self) -> bool:
        print("Connecting to {}: ".format(self.name))
        try:
            pykinect.initialize_libraries()
            self.device = pykinect.start_device(device_index=self.device_id, config=self.device_config)
            return True
        except Exception as e:
            self.device = None
            print("Failed with exception: ", e)
            return False

    def get_sensors(self):
        if not self.device:
            raise Exception(f"Not connected to {self.name}")
        
        success = False
        while not success: 
            capture = self.device.update()
            success, image = capture.get_color_image()
            timestamp = time.time()

        return {'time': timestamp, 'rgb': image}

    def close(self):
        self.device.close()
        self.device = None

    def __set_device_configuration(self):
        self.device_config = pykinect.default_configuration
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        # self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF

if __name__ == "__main__":
    rs = Azure(device_id=1)
    rs.connect()

    for i in range(50):
        img = rs.get_sensors()
        if img['rgb'] is not None:
            print("Received image{} of size:".format(i), img['rgb'].shape, flush=True)
            cv2.imshow("rgb", img['rgb'])
            cv2.waitKey(1)

        if img['rgb'] is None:
            print(img)

        time.sleep(0.1)

    rs.close()