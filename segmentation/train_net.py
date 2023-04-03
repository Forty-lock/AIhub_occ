#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

if __name__ == "__main__":
    register_coco_instances("nia_train", {}, '/dataset/occlusion/annotation/train/annotation_coco_is.json', '/dataset/occlusion/img')
    register_coco_instances("nia_test", {}, '/dataset/occlusion/annotation/test/annotation_coco_is.json', '/dataset/occlusion/img')
    my_dataset_train_metadata = MetadataCatalog.get("nia_train")
    dataset_dicts = DatasetCatalog.get("nia_train")

    cfg = get_cfg()
    add_pointrend_config(cfg)

    cfg.merge_from_file("configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.DATASETS.TRAIN = ("nia_train",)
    cfg.DATASETS.TEST = ("nia_test",)

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 82
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 82

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
