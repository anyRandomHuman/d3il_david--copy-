import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


cup_pred_class = 41
stacked_2cups_class = 39
BOX_FEATURE = "box"
MASK_FEATURE = "mask"
ALL_FEATURE = "all"


class Object_Detector:
    def __init__(
        self,
        is_joint,
        device="cuda",
        to_tensor=False,
        sam_checkpoint="sam_models/sam_vit_b.pth",
        model_type="vit_b",
    ):
        self.is_joint = is_joint
        self.to_tensor = to_tensor
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def predict(self, img):
        outputs = self.mask_generator.generate(img)
        self.prediction = sorted(
            outputs, key=(lambda x: x["stability_score"]), reverse=True
        )

    def get_box_feature(self, top_n):
        if top_n < 0:
            top_n = len(self.prediction) + top_n
        shape = tuple(self.prediction[0]["segmentation"].shape[0:2]) + (top_n,)
        features = np.zeros(shape)

        if not len(self.prediction):
            return features
        for i in range(top_n):
            ann = self.prediction[i]
            b = ann["bbox"]
            features[b[1] : b[1] + b[3], b[0] : b[0] + b[2], i] = 1

        if self.to_tensor:
            features = torch.from_numpy(features).int()
        return features

    def get_mask_feature(self, top_n):
        if top_n < 0:
            top_n = len(self.prediction) + top_n
        shape = tuple(self.prediction[0]["segmentation"].shape[0:2]) + (top_n,)
        features = np.zeros(shape)

        if not len(self.prediction):
            return features
        for i in range(top_n):
            ann = self.prediction[i]
            m = ann["segmentation"]
            features[m, i] = 1
        if self.to_tensor:
            features = torch.from_numpy(features).int()
        return features

    def joint_feature(self, features):
        joint_mask = torch.zeros(features.shape[:-1])
        if not self.to_tensor:
            features = torch.from_numpy(features)
        for i in range(features.shape[-1]):
            joint_mask = torch.logical_or(joint_mask, features[:, :, i])
        return joint_mask

    def get_feature(self, img):
        features = self.extract_feature(img)
        if self.is_joint:
            features = self.joint_feature(features)
        if self.to_tensor:
            features = torch.from_numpy(features)
        return features


if __name__ == "__main__":
    img = cv2.imread("moving_cup.jpg")

    obj_det = Object_Detector(True, to_tensor=False)
    obj_det.predict(img)
    mask = obj_det.get_mask_feature(9)
    mask = obj_det.joint_feature(mask)
    mask = np.expand_dims(mask, -1)
    mask = np.repeat(mask, 3, axis=-1)
    cv2.imshow("1", np.where(mask, img, np.zeros(img.shape)).astype(np.uint8))
    cv2.waitKey(0)
