from ultralytics import YOLO
import torch
import numpy as np


class Yolo_Detrector:
    def __init__(self, path, to_tensor, device) -> None:
        self.model = YOLO(path)
        self.to_tensor = to_tensor
        self.device = device

    def predict(self, img):
        self.img = img
        self.prediction = self.model.predict(img)

    def track(self, img):
        self.img = img
        self.prediction = self.model.track(img, persist=True)

    def get_feature(self, top_n):
        # igonre top_n, always retrive all
        p = self.prediction[0]
        num_boxes = p.boxes.shape[0]

        features = np.zeros(p.orig_shape + (num_boxes,), dtype=np.uint8)

        b = self.prediction[0].boxes.xyxy.int()

        for i in range(num_boxes):
            box = b[i]
            features[box[1] : box[3], box[0] : box[2], i] = 1

        if self.to_tensor:
            features = torch.from_numpy(features)
        return features

    def get_Bbox(self):
        box = self.prediction[0].boxes.xyxy.int()
        if self.to_tensor:
            return box
        else:
            return box.numpy(box)

    def joint_feature(self, features):
        if self.to_tensor:
            joint_mask = torch.zeros(features.shape[1:])
            for i in range(features.shape[0]):
                joint_mask = torch.logical_or(joint_mask, features[i, :, :])
        if not self.to_tensor:
            joint_mask = np.zeros(features.shape[1:])
            for i in range(features.shape[0]):
                joint_mask = np.logical_or(joint_mask, features[i, :, :])
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

    def get_img_with_segment(self):
        return self.prediction[0].plot(
            labels=False, boxes=False, conf=False, probs=False, color_mode="class"
        )


if __name__ == "__main__":
    import pathlib

    video_dir = pathlib.Path(
        "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-17_09_11/images/Azure_0"
    )
    detector = Yolo_Detrector(
        "/home/alr_admin/david/real_robot/models/yolov10n.pt", False, "cuda"
    )
    i = 0
    for img in sorted(video_dir.iterdir(), key=lambda p: int(p.name.partition(".")[0])):
        detector.track(img)
        i += 1
        if not i % 20:
            detector.prediction[0].show()
