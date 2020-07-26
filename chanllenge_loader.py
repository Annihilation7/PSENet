#  -*- coding:utf-8 -*-
#  Editor      : Pycharm
#  File        : chanllenge_loader.py
#  Created     : 2020/7/18 下午12:52
#  Description : data loader for challenge of xunfei


import json
import os
import random

import Polygon as plg
import cv2
import numpy as np
import pyclipper
import torch
import torchvision.transforms as transforms
from torch.utils import data


# =========================  transform for img  =========================

def random_scale(img, min_size):
    """
    min_size default: 640
    正因为调用了这个函数，初始化成归一化的坐标就不用维护了
    """
    h, w = img.shape[: 2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[: 2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def aug(img, bboxes):
    """bboxes是list套若干个ndarray"""
    return img

# =========================  transform for img end  =========================

# =========================  transform for imgs  =========================

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        # return i, j, th, tw
    for idx in range(len(imgs)):
        imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

# =========================  transform for imgs end  =========================


def shrink(bboxes, rate, max_shr=20):
    def dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def perimeter(bbox):
        peri = 0.0
        for i in range(bbox.shape[0]):
            peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
        return peri

    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue

        try:
            # shrink后只有一个有效instances， 属于正常情况
            shrinked_bbox_general = np.array(shrinked_bbox)[0]
            if shrinked_bbox_general.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue
            shrinked_bboxes.append(shrinked_bbox_general)
        except Exception:  # 由于gt太细，shrink后可能会有若干个instance
            for box in shrinked_bbox:
                box = np.array(box)
                if box.shape[0] <= 2:
                    shrinked_bboxes.append(box)
                    continue
                shrinked_bboxes.append(box)

    return shrinked_bboxes


class ChallengeTrainDateset(data.Dataset):
    def __init__(
            self, ann_path, img_size, kernel_num=7, min_scale=0.4
    ):
        self.img_dir = os.path.dirname(ann_path)
        self.img_size = (img_size, img_size) \
            if isinstance(img_size, int) else tuple(img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale
        self._deploy(ann_path)
        print("data prepared finished.")

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        info = self.info[index]

        img = cv2.cvtColor(cv2.imread(info["img_path"]), cv2.COLOR_BGR2RGB)
        tags = info["tags"]
        bboxes = info["bboxes"]
        # dynamic imgaug
        img_aug = aug(img, bboxes)
        h, w = img_aug.shape[:2]
        for i in range(len(bboxes)):
            bboxes[i] /= [w, h]  # normalize the bboxes
        img_aug = random_scale(img_aug, self.img_size[0])  # multi scale training

        h, w = img_aug.shape[:2]
        for i in range(len(bboxes)):
            bboxes[i] *= [w, h]  # inv-normalize

        gt_text = np.zeros(img_aug.shape[:2], dtype=np.uint8)
        training_mask = np.ones(img_aug.shape[:2], dtype=np.uint8)
        for i in range(len(bboxes)):
            cv2.drawContours(
                gt_text, [bboxes[i].astype(np.int32)], 0, i + 1, -1)
            if tags[i]:
                cv2.drawContours(
                    training_mask, [bboxes[i].astype(np.int32)], 0, 0, -1)

        # kernels
        gt_kernels = []
        for i in range(self.kernel_num):  # i=0的时候就是原gt，此后依次减小
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros(img_aug.shape[: 2], dtype=np.uint8)
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        self.imgs = [img_aug, gt_text, training_mask]
        self.imgs.extend(gt_kernels)
        self.imgs = random_crop(self.imgs, self.img_size)

        img, gt_text, training_mask, gt_kernels = \
            self.imgs[0], self.imgs[1], self.imgs[2], self.imgs[3:]
        gt_text[gt_text > 0] = 1
        # shape=[kernel_num, self.img_size[0], self.img_size[1]]
        gt_kernels = np.array(gt_kernels)

        # to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()

        return img, gt_text, gt_kernels, training_mask

    def _deploy(self, info_json):
        self.info = []

        with open(info_json, 'r', encoding="utf-8") as fp:
            info_ori = json.load(fp)
        for k, v in info_ori.items():
            if len(v) == 0:  # 图上没有bbox
                continue

            tmp_info = {}  # key --- img_path, width, height, bboxes_norm, tags

            img_name = 'img_' + k.split('_', 1)[1] + ".jpg"
            img_path = os.path.join(self.img_dir, img_name)
            assert os.path.exists(img_path), \
                "train data path: {} not exist".format(img_path)
            tmp_info["img_path"] = img_path

            tmp_gts = []
            tmp_tags = []
            for points_info in v:
                points_str = points_info["points"]
                points = [int(j) for i in points_str for j in i]
                bboxes = np.array(points, dtype=np.float32).reshape(-1, 2)
                tmp_gts.append(bboxes)
                tmp_tags.append(points_info["illegibility"])
            tmp_info["bboxes"] = tmp_gts
            tmp_info["tags"] = tmp_tags

            self.info.append(tmp_info)

    def for_vis_gt_debug(self, output_dir, random_save_num):
        random_idxes = np.random.choice(
            [i for i in range(len(self.info))], random_save_num)
        for idx, random_img_idx in enumerate(random_idxes):
            info = self.info[random_img_idx]
            save_path = os.path.join(output_dir,
                                     os.path.basename(info["img_path"]))
            img = cv2.imread(info["img_path"])
            bboxes = info["bboxes"]
            tags = info["tags"]
            for bbox, tag in zip(bboxes, tags):
                bbox = bbox.astype(np.int32)
                color = (0, 255, 0) if not tag else (0, 0, 255)
                cv2.polylines(img, [bbox], True, color, 2)  # 顺时针、逆时针都行
            cv2.imwrite(save_path, img)
            print("{}/{} completed.".format(idx + 1, random_save_num))

    def debug(self, out_dir):
        assert len(self.imgs) == 3 + self.kernel_num
        cv2.imwrite(os.path.join(out_dir, "img_ori.jpg"), self.imgs[0])
        gt_text = self.imgs[1].copy()
        gt_text[gt_text != 0] = 255
        cv2.imwrite(os.path.join(out_dir, "gt.jpg"), gt_text)
        training_mask = self.imgs[2].copy()
        training_mask[training_mask == 0] = 255
        cv2.imwrite(os.path.join(out_dir, "training_mask.jpg"), training_mask)
        for i in range(3, len(self.imgs)):
            gt_kernel = self.imgs[i].copy()
            gt_kernel[gt_kernel == 1] = 255
            cv2.imwrite(
                os.path.join(out_dir, "kernel_{}.jpg".format(i - 3)), gt_kernel)


def get_train_data_loader(
        ann_path, img_size, batch_size, num_workers,
        kernel_num, min_scale):
    dataset = ChallengeTrainDateset(
        ann_path, img_size, kernel_num, min_scale)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, pin_memory=True)


if __name__ == "__main__":
    ann_path = "/home/nofalling/Downloads/data/eval/validation.json"
    img_size = 640
    batch_size = 4
    num_workers = 1
    kernel_num = 7
    min_scale = 0.4

    train_loader = get_train_data_loader(
        ann_path, img_size, batch_size,
        num_workers, kernel_num, min_scale)

    iter_ob = iter(train_loader)
    data = next(iter_ob)
    for i in range(4):
        print(data[i].shape)
