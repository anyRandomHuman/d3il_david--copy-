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

from agents.utils.hdf5_to_img import crop_and_resize, process_cropeed

DELTA_T = 0.034

logger = logging.getLogger(__name__)


class RealRobot(BaseSim):
    def __init__(self, device: str, resizes, crops, crop_resizes, top_n, path):
        super().__init__(seed=-1, device=device)

        self.p4 = FrankaArm(
            name="p4",
            ip_address="141.3.53.154",
            port=50053,
            control_type=ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL,
            hz=100,
        )
        assert self.p4.connect(), f"Connection to {self.p4.name} failed"

        self.p4_hand = FrankaHand(name="p4_hand", ip_address="141.3.53.154", port=50054)
        assert self.p4_hand.connect(), f"Connection to {self.p4_hand.name} failed"

        self.cam0 = Azure(device_id=0)
        self.cam1 = Azure(device_id=2)
        assert self.cam0.connect(), f"Connection to {self.cam0.name} failed"
        assert self.cam1.connect(), f"Connection to {self.cam1.name} failed"

        self.i = 0
        self.top_n = top_n

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

                self.p4.reset()
                self.p4_hand.reset()

                self.create_record_dir(self.task_record_path)
                self.i = 0

        logger.info("Quitting evaluation")

        km.close()
        self.p4.close()
        self.p4_hand.reset()

    def __get_obs(self):

        img0 = self.cam0.get_sensors()["rgb"][:, :, :3]  # remove depth
        img1 = self.cam1.get_sensors()["rgb"][:, :, :3]

        crop0 = crop_and_resize(
            img0,
            tuple(self.resizes[0]),
            crop=self.crops[0],
            crop_resize=self.crop_resizes[0],
        )

        crop1 = crop_and_resize(
            img1,
            tuple(self.resizes[1]),
            crop=self.crops[1],
            crop_resize=self.crop_resizes[1],
        )

        self.detector.predict(crop0)
        f0 = self.detector.get_mask_feature()
        f0 = self.detector.joint_feature(f0)
        masked0 = self.detector.get_masked_img(f0)

        self.detector.predict(crop0)
        f1 = self.detector.get_mask_feature()
        f1 = self.detector.joint_feature(f1)
        masked1 = self.detector.get_masked_img(f1)

        self.write_obs(
            crop0,
            crop1,
            masked0,
            masked1,
            self.path,
        )

        processed_img0 = process_cropeed(crop0, to_tensor=False)
        processed_img1 = process_cropeed(crop1, to_tensor=False)

        return (processed_img0, processed_img1, masked0, masked1)

    def set_detector(self, det):
        self.detector = det

    def write_obs(self, img0, img1, masked0, masked1, path):
        cv2.imwrite(str(path / "cam0" / f"{self.i}.jpg"), img0)
        cv2.imwrite(str(path / "cam1" / f"{self.i}.jpg"), img1)
        cv2.imwrite(str(path / "mask0" / f"{self.i}.jpg"), masked0)
        cv2.imwrite(str(path / "mask1" / f"{self.i}.jpg"), masked1)

        self.i += 1

    def create_record_dir(self, path):
        t = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        record_dir = Path(path) / t
        Path.mkdir(record_dir)
        Path.mkdir(record_dir / "cam0")
        Path.mkdir(record_dir / "cam1")
        Path.mkdir(record_dir / "mask0")
        Path.mkdir(record_dir / "mask1")

        self.path = record_dir
