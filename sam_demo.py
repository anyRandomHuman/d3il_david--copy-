import numpy as np

import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys


def show_anns(anns, input):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["predicted_iou"]), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns[:5]:
        m = ann["segmentation"]
        b = ann["bbox"]
        color_mask = np.array([0, 0, 0, 0.35])
        single_mask = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        single_mask[:, :, 3] = 0
        single_mask[b[1] : b[1] + b[3], b[0] : b[0] + b[2]] = color_mask  # x,y
        # single_mask[m] = color_mask
        img[m] = color_mask
        plt.imshow(input)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        ax.imshow(single_mask)
        plt.axis("off")
        plt.show()

    # ax.imshow(img)


sam_checkpoint = "/home/alr_admin/david/praktikum/d3il_david/sam_models/sam_vit_b.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
# prompt_predictor = SamPredictor(sam)


img = cv2.imread("moving_cup.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# yellow = np.array([183, 139, 18])  # rgb
# diff = np.apply_over_axes(
#     lambda x, axes: np.isclose(x, yellow, rtol=0.1, atol=[10, 10, 10]), img, [0, 1]
# )
# y_cup_coord = np.argmin(diff, axis=-1)
# prompt_predictor.predict(point_coords=y_cup_coord)

masks = mask_generator.generate(img)


plt.figure(figsize=(20, 20))
# plt.imshow(image)
show_anns(masks, image)
# plt.axis("off")
# plt.show()
