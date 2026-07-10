import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, crop_and_resize_torch, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm


def iter_cropped_batches_multiperson(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, max_batch=16):
    assert isinstance(input_path, str)
    assert bbx_xys.ndim == 3

    person_num, frame_count, _ = bbx_xys.shape
    chunk_frames = max(1, int(max_batch) // max(1, person_num))
    bbx_xys_np = bbx_xys.detach().cpu().numpy() if isinstance(bbx_xys, torch.Tensor) else np.asarray(bbx_xys)

    cap = cv2.VideoCapture(input_path)
    frame_cursor = 0
    while frame_cursor < frame_count:
        frames = []
        start = frame_cursor
        for _ in range(min(chunk_frames, frame_count - start)):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            if img_ds != 1.0:
                frame_rgb = cv2.resize(frame_rgb, (0, 0), fx=img_ds, fy=img_ds)
            frames.append(frame_rgb)
            frame_cursor += 1

        if not frames:
            break

        crops = []
        crop_bbx = []
        for person_id in range(person_num):
            for local_idx, frame_rgb in enumerate(frames):
                frame_idx = start + local_idx
                center = bbx_xys_np[person_id, frame_idx, :2] * img_ds
                size = bbx_xys_np[person_id, frame_idx, 2] * img_ds
                crop, bbx_final = crop_and_resize(
                    frame_rgb,
                    center,
                    size,
                    dst_size=img_dst_size,
                    enlarge_ratio=1.0,
                )
                crops.append(crop)
                crop_bbx.append(bbx_final / img_ds)

        imgs = torch.from_numpy(np.stack(crops))
        bbx_batch = torch.from_numpy(np.stack(crop_bbx)).float()
        imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2).contiguous()
        yield imgs, bbx_batch, start, frame_cursor

    cap.release()


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256)
    return imgs, bbx_xys


def get_batch_multiperson(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    # Similar to get_batch but handling multiple persons
    # Assume bbx_xys is a tensor of shape (person_num, F, 3)

    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0
        imgs = input_path
    
    person_num, F, _ = bbx_xys.shape
    imgs_list = []
    bbx_xys_ds_list = []

    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().cuda()  # (F, 3, H, W)
    bbx_xys = bbx_xys.cuda()
    for person_id in range(person_num):
        img, bbx_xys_ds = crop_and_resize_torch(
            imgs,
            bbx_xys[person_id, :, :2] * img_ds,
            bbx_xys[person_id, :, 2] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)

    # B = person_num * F
    imgs = torch.cat(imgs_list, dim=0)  # (B, 3, 256, 256)
    bbx_xys = torch.cat(bbx_xys_ds_list, dim=0) / img_ds  # (B, 3)

    mean, std = IMAGE_MEAN[None, :, None, None].cuda(), IMAGE_STD[None, :, None, None].cuda()
    imgs = ((imgs / 255.0 - mean) / std)
    return imgs, bbx_xys.cpu()


class Extractor:
    def __init__(self, tqdm_leave=True, batch_size=16):
        self.extractor: HMR2 = load_hmr2().cuda().eval()
        self.tqdm_leave = tqdm_leave
        self.batch_size = batch_size

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            if bbx_xys.ndim == 3:  # multiple persons (person_num, F, 3)
                imgs, bbx_xys = get_batch_multiperson(video_path, bbx_xys, img_ds=img_ds)
            else:
                imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        batch_size = self.batch_size  # 5GB GPU memory, occupies all CUDA cores of 3090
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size].cuda()

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features

    def extract_video_features_multiperson(self, video_path, bbx_xys, img_ds=0.5):
        person_num, F, _ = bbx_xys.shape
        if isinstance(video_path, str):
            features_by_person = [[] for _ in range(person_num)]
            for imgs, _, start, end in iter_cropped_batches_multiperson(
                video_path,
                bbx_xys,
                img_ds=img_ds,
                max_batch=self.batch_size,
            ):
                n = end - start
                imgs = imgs.cuda()
                with torch.no_grad():
                    feature = self.extractor({"img": imgs}).detach().cpu()
                for person_id in range(person_num):
                    features_by_person[person_id].append(feature[person_id * n : (person_id + 1) * n])
            return torch.stack([torch.cat(parts, dim=0) for parts in features_by_person], dim=0)

        features = self.extract_video_features(video_path, bbx_xys, img_ds=img_ds)
        return features.reshape(person_num, F, 1024)  # (person_num, F, 1024)
