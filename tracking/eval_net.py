#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import glob
import os
import json

import numpy as np
from tqdm import tqdm
import random
from tqdm import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from tridentnet import add_tridentnet_config
from detectron2.utils.visualizer import Visualizer, ColorMode

if __name__ == "__main__":

    cfg = get_cfg()
    add_tridentnet_config(cfg)

    cfg.merge_from_file("configs/tridentnet_fast_R_101_C4_3x.yaml")
    cfg.DATASETS.TEST = ("nia_test",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 82
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    images = glob.glob('/dataset/occlusion/annotation/test/*.json')

    aps = []
    st = ''

    for idx, i in enumerate(tqdm(images)):
        register_coco_instances("nia_test_%04d" % idx, {}, i, '/dataset/occlusion/img')

        with open(i, 'r') as f:
            json_anno = json.load(f)

        name = json_anno['images'][0]['file_name'][:-9]

        evaluator = COCOEvaluator("nia_test_%04d" % idx, output_dir="./output")
        val_loader = build_detection_test_loader(cfg, "nia_test_%04d" % idx)
        temp = inference_on_dataset(predictor.model, val_loader, evaluator)

        if np.isnan(temp['bbox']['AP']):
            temp['bbox']['AP'] = 0

        aps.append(temp['bbox']['AP'])

        st = st + '%s\t%.4f\n' % (name, temp['bbox']['AP'])

    print(np.mean(aps))

    with open('./track_map.txt', 'a+') as f:
        f.write(st)