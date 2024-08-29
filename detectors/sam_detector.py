# %%

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
        device="cuda",
        to_tensor=False,
        path="/home/alr_admin/david/praktikum/d3il_david/detector_models/sam_vit_b.pth",
        model_type="vit_b",
        sort="predicted_iou",
    ):
        self.to_tensor = to_tensor
        self.sam = sam_model_registry[model_type](checkpoint=path)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.sort = sort

    def predict(self, img):
        self.img = img
        outputs = self.mask_generator.generate(img)
        self.prediction = sorted(outputs, key=(lambda x: x[self.sort]), reverse=True)

    def get_box_feature(self, top_n):
        shape = tuple(self.prediction[0]["segmentation"].shape[0:2]) + (top_n,)
        features = np.zeros(shape)

        if not len(self.prediction):
            return features
        for i in range(top_n):
            ann = self.prediction[i]

            if ann["area"] > 1500 or ann["area"] < 100:
                continue
            b = ann["bbox"]
            for j in range(len(b)):
                b[j] = int(b[j])
            features[b[1] : b[1] + b[3], b[0] : b[0] + b[2], i] = 1

        if self.to_tensor:
            features = torch.from_numpy(features).int()
        return features

    def get_feature(self, top_n):
        if top_n == -1:
            top_n = len(self.prediction)
        shape = tuple(self.prediction[0]["segmentation"].shape[0:2]) + (top_n,)
        features = np.zeros(shape)

        if not len(self.prediction):
            return features
        found = 0
        j = 0
        while found < top_n and j < len(self.prediction):
            ann = self.prediction[j]
            j += 1
            if ann["area"] > 2500 or ann["area"] < 50:
                continue
            found += 1
            m = ann["segmentation"]
            features[m, found - 1] = 1
        if self.to_tensor:
            features = torch.from_numpy(features).int()
        return features

    def joint_feature(self, features):
        if self.to_tensor:
            joint_mask = torch.zeros(features.shape[:-1])
            for i in range(features.shape[-1]):
                joint_mask = torch.logical_or(joint_mask, features[:, :, i])
        else:
            joint_mask = np.zeros(features.shape[:-1])
            for i in range(features.shape[-1]):
                joint_mask = np.logical_or(joint_mask, features[:, :, i])
        return joint_mask

    def get_masked_img(self, feature):
        if self.to_tensor:
            feature = torch.from_numpy(feature)
        img = np.where(
            np.expand_dims(feature, -1).repeat(3, -1),
            self.img,
            np.zeros(self.img.shape),
        )
        return img


if __name__ == "__main__":

    img = cv2.imread("0.png")

    iou = "predicted_iou"
    stability = "stability_score"
    obj_det = Object_Detector(device="cuda", to_tensor=False, sort=iou)

    mask = obj_det.predict(img)
    mask = obj_det.get_mask_feature(-1)
    mask = obj_det.joint_feature(mask)
    img = obj_det.get_masked_img(mask)

    cv2.imshow("1", img)
    cv2.waitKey(0)
