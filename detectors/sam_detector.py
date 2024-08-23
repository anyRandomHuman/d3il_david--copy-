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
        feature_type,
        device="cuda",
        to_tensor=False,
        sam_checkpoint="/home/alr_admin/david/praktikum/d3il_david/sam_models/sam_vit_b.pth",
        model_type="vit_b",
    ):
        self.is_joint = is_joint
        self.f_type = feature_type
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

    @staticmethod
    def joint_feature(features):
        joint_mask = torch.zeros(features.shape[:-1])
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

    obj_det = Object_Detector(True, MASK_FEATURE, obj_classes=[41, 39], to_tensor=False)

    mask = obj_det.get_feature(img)
    mask = np.expand_dims(mask, -1)
    mask = np.repeat(mask, 3, axis=-1)
    cv2.imshow("1", np.where(mask, img, np.zeros(img.shape)))
    cv2.waitKey(0)
