import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor


class Sam2:
    def __init__(
        self,
        model_cfg="sam2_hiera_l.yaml",
        sam2_checkpoint="detector_models/sam2_hiera_large.pt",
        device="cuda",
    ) -> None:
        self.predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )
        self.id = 0

    def predict(self, path):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor.init_state(path)

            video_segments = (
                {}
            )  # video_segments contains the per-frame segmentation results
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

    def predict_video(self):

        self.video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().squeeze().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def add_boxes(self, boxes, frame_idx):
        for i in len(boxes):
            _, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                boxes=boxes[i],
                frame_idx=frame_idx,
                obj_id=self.id,
            )
            self.id += 1

    def add_all_points(self, all_points: list, all_labels: np.ndarray, frame_idx):
        for i in len(all_points):
            _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=frame_idx,
                obj_id=self.id,
                points=all_points[i],
                labels=all_labels[i],
            )
            self.id += 1

    def init_states(self, path):
        self.path = path
        self.state = self.predictor.init_state(path)
        self.id = 0
        return self.state

    def get_feature(self):
        joint_masks = []
        for out_frame_idx in range(len(self.video_segments.keys())):

            joint_mask = np.zeros(
                self.video_segments[out_frame_idx].shape, dtype=np.uint8
            )

            for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                joint_mask = np.logical_or(joint_mask, out_mask)
            joint_masks.append(joint_mask)
        return joint_masks

    def joint_feature(self, features):
        if self.to_tensor:
            joint_mask = torch.zeros(features.shape[:-1])
            for i in range(features.shape[-1]):
                joint_mask = torch.logical_or(joint_mask, features[:, :, i])
        if not self.to_tensor:
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

        return img.astype(np.uint8)
