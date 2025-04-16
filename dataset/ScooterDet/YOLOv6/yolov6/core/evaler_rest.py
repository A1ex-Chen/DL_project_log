#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import yaml
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER, NCOLS
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.general import download_ckpt
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.utils.torch_utils import time_sync, get_model_info


class Evaler:










    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod




        context, bindings, binding_addrs, trt_batch_size = init_engine(engine)
        assert trt_batch_size >= self.batch_size, f'The batch size you set is {self.batch_size}, it must <= tensorrt binding batch size {trt_batch_size}.'
        tmp = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        # warm up for 10 times
        for _ in range(10):
            binding_addrs['images'] = int(tmp.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
        dataloader = init_data(None,'val')
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc="Inferencing model in validation dataset.", ncols=NCOLS)
        for imgs, targets, paths, shapes in pbar:
            nb_img = imgs.shape[0]
            if nb_img != self.batch_size:
                # pad to tensorrt model setted batch size
                zeros = torch.zeros(self.batch_size - nb_img, 3, *imgs.shape[2:])
                imgs = torch.cat([imgs, zeros],0)
            t1 = time_sync()
            imgs = imgs.to(self.device, non_blocking=True)
            # preprocess
            imgs = imgs.float()
            imgs /= 255

            self.speed_result[1] += time_sync() - t1  # pre-process time

            # inference
            t2 = time_sync()
            binding_addrs['images'] = int(imgs.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
            # in the last batch, the nb_img may less than the batch size, so we need to fetch the valid detect results by [:nb_img]
            nums = bindings['num_dets'].data[:nb_img]
            boxes = bindings['det_boxes'].data[:nb_img]
            scores = bindings['det_scores'].data[:nb_img]
            classes = bindings['det_classes'].data[:nb_img]
            self.speed_result[2] += time_sync() - t2  # inference time

            self.speed_result[3] += 0
            pred_results.extend(convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, self.ids))
            self.speed_result[0] += self.batch_size
        return dataloader, pred_results