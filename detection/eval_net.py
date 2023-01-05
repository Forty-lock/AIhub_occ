#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import glob
import os
import cv2
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
    register_coco_instances("nia_test", {}, './annotation_coco.json', 'E:/dataset/occlusion/img')

    cfg = get_cfg()
    add_tridentnet_config(cfg)

    cfg.merge_from_file("configs/tridentnet_fast_R_101_C4_3x.yaml")
    cfg.DATASETS.TEST = ("nia_test",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 82
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("nia_test", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "nia_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    dataset_dicts = DatasetCatalog.get("nia_test")
    for idx, d in enumerate(tqdm(random.sample(dataset_dicts, 100))):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im, MetadataCatalog.get("nia_test"), scale=1.2, instance_mode=ColorMode.IMAGE)
        point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

        cv2.imwrite('./results/img_%05d_%s.png' % (idx, d["file_name"].split('\\')[-1].split('.')[0]),
                    point_rend_result)