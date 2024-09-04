import numpy as np

import matplotlib.pyplot as plt

import cv2
import torch

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import sys


# def show_anns(anns, input):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x["predicted_iou"]), reverse=True)
#     # ax = plt.gca()
#     # ax.set_autoscale_on(False)

#     img = np.ones(
#         (
#             sorted_anns[0]["segmentation"].shape[0],
#             sorted_anns[0]["segmentation"].shape[1],
#             4,
#         )
#     )
#     img[:, :, 3] = 0
#     for ann in sorted_anns[:5]:
#         m = ann["segmentation"]
#         b = ann["bbox"]
#         color_mask = np.array([0, 0, 0, 0.35])
#         single_mask = np.ones(
#             (
#                 sorted_anns[0]["segmentation"].shape[0],
#                 sorted_anns[0]["segmentation"].shape[1],
#                 4,
#             )
#         )
#         single_mask[:, :, 3] = 0
#         single_mask[b[1] : b[1] + b[3], b[0] : b[0] + b[2]] = color_mask  # x,y
#         # single_mask[m] = color_mask
#         img[m] = color_mask
#         plt.imshow(input)
#         ax = plt.gca()
#         ax.set_autoscale_on(False)
#         ax.imshow(single_mask)
#         plt.axis("off")
#         plt.show()

#     # ax.imshow(img)


# img = cv2.imread(
#     "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-13_22_36/images/Azure_0/0.png"
# )
# plt.imshow(img)
# plt.show()

import torch
import os
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

# build_sam2_video_predictor_hf


banana = np.array([[223, 388]], dtype=np.float32)
carrot = np.array([[118, 417]], dtype=np.float32)

points = np.array([banana, carrot], dtype=np.float32)
labels = np.array([1], np.int32)
stride = 1

sam2_checkpoint = "detector_models/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
path = (
    "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-13_22_36/images/test1"
)
# outpath = (
#     "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-13_22_36/images/test"
# )
# i = 0
# for img_name in os.listdir(path):
#     img = cv2.imread(path + f"/{img_name}")
#     cv2.imwrite(outpath + f"{i}.jpg", img)
#     i += 1


# def show_mask(mask, ax, obj_id=None, random_color=False):
#     color = np.ones((1, 1, 3, ))
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color
#     ax.imshow(mask_image)


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
# predictor = build_sam2_video_predictor_hf("facebook/sam2-hiera-large")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(path)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=state, points=carrot, labels=labels, frame_idx=0, obj_id=1
    )
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=state, points=banana, labels=labels, frame_idx=0, obj_id=2
    )
    frames = os.listdir(path)

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx in range(0, len(os.listdir(path)), stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")

        img = cv2.imread(path + f"/{out_frame_idx}.jpg")

        joint_mask = np.zeros(img.shape, dtype=np.uint8)

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            joint_mask = np.logical_or(
                joint_mask, np.permute_dims(out_mask, (1, 2, 0)).repeat(3, -1)
            )

        img = np.where(joint_mask, img, joint_mask)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    plt.show()
