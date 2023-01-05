#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import cv2
import glob
import random
from tqdm import tqdm

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset,
)
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":
    register_coco_instances("nia_test", {}, './annotation_coco_is.json', 'E:/dataset/occlusion/img')

    cfg = get_cfg()
    add_pointrend_config(cfg)

    cfg.merge_from_file("configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.DATASETS.TEST = ("nia_test",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 82
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 82

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("nia_test", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "nia_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    dataset_dicts = DatasetCatalog.get("nia_test")
    for idx, d in enumerate(tqdm(random.sample(dataset_dicts, 100))):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        meta = MetadataCatalog.get("nia_test")
        ins = outputs["instances"].to("cpu")
        #
        # ins = ins[ins.pred_classes == 77]
        ins = ins[ins.pred_classes > 22]

        v = Visualizer(im, meta, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
        point_rend_result = v.draw_instance_predictions(ins).get_image()
        # point_rend_result = v.draw_dataset_dict(d).get_image()

        cv2.imwrite('./results/img_%05d_%s.png' % (idx, d["file_name"].split('\\')[-1].split('.')[0]),
                    point_rend_result)
