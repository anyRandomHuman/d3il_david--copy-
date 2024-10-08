from pathlib import Path

import matplotlib.axis
from simulation.base_sim import BaseSim
import logging
import numpy as np

from real_robot_env.robot.hardware_azure import Azure
from real_robot_env.robot.hardware_franka import FrankaArm, ControlType
from real_robot_env.robot.hardware_frankahand import FrankaHand
from real_robot_env.robot.utils.keyboard import KeyManager

import cv2
import time
import datetime
from pathlib import Path

from agents.utils.hdf5_to_img import preprocess_img

DELTA_T = 0.034

logger = logging.getLogger(__name__)


class RealRobot():
    def __init__(self, device: str, resizes, crops, crop_resizes, path):

        self.cam0 = Azure(device_id=0)
        self.cam1 = Azure(device_id=2)
        assert self.cam0.connect(), f"Connection to {self.cam0.name} failed"
        assert self.cam1.connect(), f"Connection to {self.cam1.name} failed"

        self.i = 0

        self.resizes = resizes
        # crop in the order of y, x
        self.crops = crops
        self.crop_resizes = crop_resizes

        self.task_record_path = path
        self.create_record_dir(path)

    def test_agent(self, agent):
        logger.info("Starting trained model evaluation on real robot")

        km = KeyManager()

        while km.key != "q":
            print("Press 's' to start a new evaluation, or 'q' to quit")
            km.pool()

            while km.key not in ["s", "q"]:
                km.pool()

            if km.key == "s":
                agent.reset()

                print("Starting evaluation. Press 'd' to stop current evaluation")

                km.pool()
                while km.key != "d":
                    km.pool()

                    obs = self.__get_obs()
                    pred_action = agent.predict(obs, if_vision=True).squeeze()

                    pred_joint_pos = pred_action[:7]
                    pred_gripper_command = pred_action[-1]

                    pred_gripper_command = 1 if pred_gripper_command > 0 else -1

                    self.p4.go_to_within_limits(goal=pred_joint_pos)
                    self.p4_hand.apply_commands(width=pred_gripper_command)
                    time.sleep(DELTA_T)

                logger.info("Evaluation done. Resetting robots")
                # time.sleep(1)

                self.create_record_dir(self.task_record_path)

        logger.info("Quitting evaluation")

        km.close()
        self.p4.close()
        self.p4_hand.reset()

    def __get_obs(self):

        img0 = self.cam0.get_sensors()["rgb"][:, :, :3]  # remove depth
        img1 = self.cam1.get_sensors()["rgb"][:, :, :3]

        img0_write = cv2.resize(img0, (512, 512))[:, 100:370]
        img1_write = cv2.resize(img1, (512, 512))[:, 100:370]

        self.write_obs(img0_write, img1_write, self.path)

        # cv2.imshow('0', img0)
        # cv2.imshow('1', img1)
        # cv2.waitKey(0)
        processed_img0 = preprocess_img(
            img0,
            tuple(self.resizes[0]),
            to_tensor=False,
            crop=self.crops[0],
            crop_resize=self.crop_resizes[1],
        )

        processed_img1 = preprocess_img(
            img1,
            tuple(self.resizes[1]),
            to_tensor=False,
            crop=self.crops[1],
            crop_resize=self.crop_resizes[1],
        )

        # cv2.imwrite(
        #     f"/media/alr_admin/ECB69036B69002EE/inference_record/cam0_{self.i}.png",
        #     processed_img0.transpose(1, 2, 0) * 255,
        # )
        # cv2.imwrite(
        #     f"/media/alr_admin/ECB69036B69002EE/inference_record/cam1_{self.i}.png",
        #     processed_img1.transpose(1, 2, 0) * 255,
        # )
        # self.i += 1

        return (processed_img0, processed_img1)

    def write_obs(self, img0, img1, path):
        cv2.imwrite(str(path / "cam0" / f"{self.i}.jpg"), img0)
        cv2.imwrite(str(path / "cam1" / f"{self.i}.jpg"), img1)
        self.i += 1

    def create_record_dir(self, path):
        t = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        record_dir = Path(path) / t
        Path.mkdir(record_dir)
        Path.mkdir(record_dir / "cam0")
        Path.mkdir(record_dir / "cam1")
        self.path = record_dir
