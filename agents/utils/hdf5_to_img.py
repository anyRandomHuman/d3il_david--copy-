import h5py
import os
import cv2
import torch


def read_img_from_hdf5(
    path,
    start,
    end,
    cam_resizes=[(256, 256), (256, 256)],
    device="cuda",
    to_tensor=True,
):
    """
    path =path to traj dir
    """
    f = h5py.File(path, "r")
    cams = []
    for i, cam in enumerate(list(f.keys())):
        arr = f[cam][start:end]
        imgs = []
        for img in arr:

            nparr = cv2.imdecode(img, 1)

            processed = preprocess_img(
                nparr, resize=cam_resizes[i], device=device, to_tensor=to_tensor
            )

            imgs.append(processed)
        if to_tensor:
            imgs = torch.concatenate(imgs, dim=0)
        else:
            pass
        cams.append(imgs)
    f.close()
    return cams


def preprocess_img(
    img,
    resize=(256, 256),
    crop_resize=(256, 256),
    device="cuda",
    to_tensor=True,
    crop=[0, None, 0, None],
):

    img = cv2.resize(img, crop_resize)
    img = img[crop[0] : crop[1], crop[2] : crop[3]]
    img = cv2.resize(img, resize)

    img = img.transpose((2, 0, 1)) / 255.0

    if to_tensor:
        img = torch.from_numpy(img).to(device).float().unsqueeze(0)

    return img


def read_feature(path, feature, start, end, cam_resizes, device, to_tensor=True):
    """
    return list of tensors
    len(l) = num_cam
    shape of tensor: [h,w,top_n(objects)]
    """
    path = os.path.join(path, feature)
    f = torch.load(path)
    slices = []
    for i in range(f.shape[0]):
        slice = torch.reshape(f[i][start:end], cam_resizes[i] + (-1,)).to(device)
        slices.append(slice)
    return slices


def preprocess_img_no_crop(
    img,
    resize=(256, 256),
    device="cuda",
    to_tensor=True,
):
    img = cv2.resize(img, resize)

    img = img.transpose((2, 0, 1)) / 255.0

    if to_tensor:
        img = torch.from_numpy(img).to(device).float().unsqueeze(0)

    return img


def crop_and_resize(
    img,
    resize=(256, 256),
    crop_resize=(256, 256),
    crop=[0, None, 0, None],
):
    img = cv2.resize(img, crop_resize)
    img = img[crop[0] : crop[1], crop[2] : crop[3]]
    img = cv2.resize(img, resize)
    return img


def process_cropeed(
    img,
    device="cuda",
    to_tensor=True,
):
    img = img.transpose((2, 0, 1)) / 255.0

    if to_tensor:
        img = torch.from_numpy(img).to(device).float().unsqueeze(0)

    return img


if __name__ == "__main__":
    path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_space_hdf5/insertion/2024_07_04-16_02_30"

    cam0, cam1 = read_img_from_hdf5(
        os.path.join(path, "imgs.hdf5"), 0, 1, [256, 256], "cpu"
    )
