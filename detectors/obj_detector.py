import torch
import numpy as np
from abc import ABC, abstractmethod


class Object_Detector(ABC):
    @abstractmethod
    def __init__(self, path, to_tensor=False,
        device="cuda",) -> None: 
        self.to_tensor = to_tensor
        self.device = device

    @abstractmethod
    def predict(self, img): 
        self.input = img

    @abstractmethod
    def get_box_feature(self): ...
    """
    should return shape of [h, w, num_obj]
    """

    @abstractmethod
    def get_mask_feature(self): ...

    def joint_feature(self, features):
        if self.to_tensor:
            joint_mask = torch.zeros(features.shape[:-1]).to(self.device)
            for i in range(features.shape[-1]):
                joint_mask = torch.logical_or(joint_mask, features[:, :, i])
        else:
            joint_mask = np.zeros(features.shape[:-1])
            for i in range(features.shape[-1]):
                joint_mask = np.logical_or(joint_mask, features[:, :, i])
        return joint_mask

    def get_masked_img(self, feature):
        # return self.prediction[0].plot(
        #     labels=False, boxes=False, conf=False, probs=False, color_mode="class"
        # )

        if self.to_tensor:
            img = torch.where(
            torch.unsqueeze(feature, -1).repeat_interleave(3, -1),
            torch.from_numpy(self.input).to(self.device),
            torch.zeros(self.input.shape).to(self.device),
        )
        else:
            img = np.where(
                np.expand_dims(feature, -1).repeat(3, -1),
                self.input,
                np.zeros(self.input.shape),
            )
        return img
