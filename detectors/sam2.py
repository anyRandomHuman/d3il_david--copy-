from sam2.build_sam import build_sam2_video_predictor
import torch
import os

class sam2:
    def __init__(self,
        video_path,
        device="cuda",
        to_tensor=False,
        model_checkpoint="/home/alr_admin/david/praktikum/d3il_david/detector_models/sam_vit_b.pth",
        model_type="vit_b",
        sort="predicted_iou",
        model_cfg = "sam2_hiera_l.yaml",
        ) -> None:
        self.to_tensor = to_tensor
        self.predictor = build_sam2_video_predictor(model_cfg, model_checkpoint, device=device)
        
        self.video_path = video_path


    def predict_video(self, path, stride):
        self.inference_state = self.predictor.init_state(video_path=path)
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        to_return = {}
        for out_frame_idx in range(0, len(os.listdir(path)), stride):
            

    def add_point(self, point):