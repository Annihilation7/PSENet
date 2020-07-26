#  -*- coding:utf-8 -*-
#  Editor      : Pycharm
#  File        : my_eval.py
#  Created     : 2020/7/22 下午11:55
#  Description : eval script for xunfei


import torch
import cv2
import numpy as np
from torchvision import transforms
from pse import pse


class EvalModel:
    def __init__(self, args):
        self.args = args
        self.max_size = 2240
        self.min_ratio = 1.5
        self.device = torch.device("cuda:0") \
            if args.use_gpu else torch.device("cpu")

    def inference(self, model, image):
        model.eval()
        image_preprocessed, scale = self._preprocess_image(image)
        image_preprocessed = image_preprocessed.to(self.device)
        with torch.no_grad():
            outputs = model(image_preprocessed)
            score = torch.sigmoid(outputs[:, 0, ...])
            outputs = (torch.sign(outputs - self.args.binary_th) + 1) / 2
            text = outputs[:, 0, ...]
            kernels = outputs[:, 0: self.args.kernel_num, ...] * text

            score = score.data.cpu().numpy()[0].astype(np.float32)
            text = text.data.cpu().numpy()[0].astype(np.uint8)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

            pred = pse(kernels, self.args.min_kernel_area /
                       (self.args.scale * self.args.scale))
            label_num = np.max(pred) + 1


    def _preprocess_image(self, image):
        """cv bgr img to torch preprocessed tensor"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        ratio = min(self.min_ratio, self.max_size / max(h, w))
        target_h, target_w = int(ratio * h), int(ratio * w)
        image_resize = cv2.resize(
            image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        target_h32, target_w32 = target_h, target_w
        if target_h32 % 32 != 0:
            target_h32 = target_h32 + (32 - target_h32 % 32)
        if target_w32 % 32 != 0:
            target_w32 = target_w32 + (32 - target_w32 % 32)
        resized = np.zeros((target_h32, target_w32, 3), dtype=np.uint8)
        resized[:target_h, :target_w, :] = image_resize
        # to tensor and transform
        pre_img = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )(resized)
        return pre_img, ratio

