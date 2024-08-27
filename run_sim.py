import os
import logging

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from agents.utils.sim_path import sim_framework_path
from agents.utils.hdf5_to_img import read_img_from_hdf5, preprocess_img
import torch
import matplotlib.pyplot as plt
import cv2

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


def test_agent_on_train_data(path, agent, feature_path=None, resize=(128, 256)):
    joint_poses = torch.load(path + "/leader_joint_pos.pt")
    if os.path.exists(os.path.join(path, "imgs.hdf5")):
        imgs = read_img_from_hdf5(
            os.path.join(path, "imgs.hdf5"), 0, -1, to_tensor=False
        )
    else:
        imgs = []
        path = os.path.join(path, "images")
        for cam in os.listdir(path):
            if cam == "Azure_0" or cam == "Azure_1":
                cams = []
                cam_path = os.path.join(path, cam)
                for img in os.listdir(cam_path):
                    img_path = os.path.join(cam_path, img)
                    img = cv2.imread(img_path)
                    img = preprocess_img(img, resize, to_tensor=False)
                    cams.append(img)
                imgs.append(cams)

    if feature_path:
        features = read_img_from_hdf5(
            os.path.join(feature_path, "masked_imgs.hdf5"),
            0,
            -1,
            to_tensor=False,
            cam_resizes=[resize, resize],
        )
        state_pairs = list(zip(imgs[0], imgs[1], features[0], features[1], joint_poses))
    else:
        state_pairs = list(zip(imgs[0], imgs[1], joint_poses))

    num_action = len(joint_poses) - 1
    pred_joint_poses = np.zeros((num_action, 7))

    for i in range(num_action):
        state_pair = state_pairs[i]
        obs = state_pair[:-1]
        pred_action = agent.predict(obs, if_vision=True).squeeze()
        pred_joint_pos = pred_action[:7]
        pred_gripper_command = pred_action[-1]
        pred_joint_poses[i] = pred_joint_pos

    fig, axises = plt.subplots(7)
    axises.legend()
    for i in range(7):
        ax = axises[i]
        ax.plot(
            range(num_action), pred_joint_poses[::, i], label="prediction"
        )  # Plot some data on the Axes.
        ax.plot(range(num_action), joint_poses[:-1, i], label="truth")
        ax.legend()
    plt.show()


# @hydra.main(config_path="configs", config_name="real_robot_config.yaml")
@hydra.main(config_path="configs", config_name="oc_pick_placing_config.yaml")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config,
    )

    # cfg.task_suite = "cupStacking"
    cfg.if_sim = True
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(
        "/home/alr_admin/david/praktikum/d3il_david/weights",
        sv_name="pickPlacing_oc_100data_100epoch.pth",
    )

    oc = True

    # env_sim = hydra.utils.instantiate(cfg.simulation)

    # if oc:
    #     det = hydra.utils.instantiate(cfg.detectors)
    #     env_sim.set_detector(det)

    # env_sim.test_agent(agent)

    path = "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-13_22_36"
    feature_path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_new_hdf5_downsampled/pickPlacing_features/2024_08_05-13_22_36"
    test_agent_on_train_data(path, agent, feature_path=feature_path, resize=(128, 256))

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()
