import logging
from environments.dataset.base_dataset import TrajectoryDataset
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
import h5py


def img_file_key(p: Path):
    return int(p.name.partition(".")[0])


class Hdf5_Dataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        task_suite: str = "cupStacking",
        device="cpu",
        obs_dim: int = 20,
        action_dim: int = 2,
        max_len_data: int = 256,
        window_size: int = 1,
        if_sim: bool = False,
        cam_0_w=256,
        cam_0_h=256,
        cam_1_w=256,
        cam_1_h=256,
        cam_num=2,
    ):

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        logging.info("Loading Real Robot Dataset")

        actions = []
        masks = []

        if task_suite == "cupStacking":
            data_dir = Path(data_directory + "/cupstacking")
        elif task_suite == "pickPlacing":
            data_dir = Path(data_directory + "/banana")
        elif task_suite == "insertion":
            data_dir = Path(data_directory + "/insertion")
        else:
            raise ValueError("Wrong name of task suite.")

        if not if_sim:
            load_img = -1
        else:
            load_img = 1

        self.cam_0_resize = [cam_0_w, cam_0_h]
        self.cam_1_resize = [cam_1_w, cam_1_h]

        self.imgs = []
        for traj_dir in tqdm(data_dir.iterdir()):
            traj_img = []
            image_path = traj_dir / "images"
            image_hdf5 = traj_dir / "imgs.hdf5"
            if Path(image_path).is_dir():
                pass
            elif Path(image_hdf5).exists():
                with h5py.File(image_hdf5, "r") as f:
                    for cam in list(f.keys())[:cam_num]:
                        cam_img = []
                        for img_code in f[cam]:
                            img = cv2.imdecode(img_code, 1)
                            img = img.transpose((2, 0, 1)) / 255.0
                            img = (
                                torch.from_numpy(img)
                                .to(self.device)
                                .float()
                                .unsqueeze(0)
                            )
                            # TODO:resize if necessary
                            cam_img.append(img)
                        traj_img.append(cam_img)
            self.imgs.append(traj_img)

            zero_action = torch.zeros(
                (1, self.max_len_data, self.action_dim), dtype=torch.float32
            )
            zero_mask = torch.zeros((1, self.max_len_data), dtype=torch.float32)

            joint_pos = torch.load(traj_dir / "follower_joint_pos.pt")
            # joint_vel = torch.load(traj_dir / "joint_vel.pt")
            gripper_command = torch.load(traj_dir / "follower_gripper_state.pt")

            valid_len = len(joint_pos) - 1

            # zero_action[0, :valid_len, :] = torch.cat(
            #     [joint_pos, joint_vel, gripper_command[:, None]], dim=1
            # )

            zero_action[0, :valid_len, :] = torch.cat(
                [joint_pos[1:], gripper_command[1:, None]], dim=1
            )

            zero_mask[0, :valid_len] = 1

            actions.append(zero_action)
            masks.append(zero_mask)

        self.actions = torch.cat(actions).to(device).float()
        self.masks = torch.cat(masks).to(device).float()

        self.num_data = len(self.actions)

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={self.window_size}"
                )
            else:
                slices += [
                    (i, start, start + self.window_size)
                    for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        img_0 = self.imgs[0][i][start:end]
        img_1 = self.imgs[1][i][start:end]
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        # bp_imgs = self.bp_cam_imgs[i][start:end]
        # inhand_imgs = self.inhand_cam_imgs[i][start:end]

        return img_0, img_1, act, mask
