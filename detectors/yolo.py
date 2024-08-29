from ultralytics import YOLO
from detectors.obj_detector import Object_Detector
import torch
import numpy as np


class Yolo_Detrector(Object_Detector):
    def __init__(self, path, to_tensor, device) -> None:
        super().__init__(path)
        self.model = YOLO(path)
        self.to_tensor = to_tensor
        self.device = device

    def predict(self, img):
        self.prediction = self.model.predict(img)

    def get_box_feature(self):
        p = self.prediction[0]
        num_boxes = p.boxes.shape[0]

        features = torch.zeros(p.orig_shape[:-1] + (num_boxes,))

        b = self.prediction[0].boxes.xyxy.int()

        for i in range(num_boxes):
            box = b[i]
            features[box[1] : box[3], box[0] : box[2], i] = 1

        if self.to_tensor:
            features = torch.from_numpy(features).int()
        return features

    def get_Bbox(self):
        return self.prediction[0].boxes.xyxy.int()

    @staticmethod
    def joint_feature(features):
        joint_mask = torch.zeros(features.shape[:-1])
        for i in range(features.shape[-1]):
            joint_mask = torch.logical_or(
                joint_mask, torch.from_numpy(features[:, :, i])
            )
        return joint_mask

    def get_masked_img(self, feature):
        img = (
            torch.where(
                torch.unsqueeze(feature, -1).repeat_interleave(3, -1),
                torch.from_numpy(self.img),
                torch.zeros(self.img.shape),
            )
            .int()
            .numpy()
        )
        return img.astype(np.uint8)
