from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import base64
import io

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
#logging.getLogger('detectron2').setLevel(logging.WARNING)

# import some common libraries
import numpy as np
import os, json, cv2, random
import copy
import matplotlib.pyplot as plt
# import imutils
# from utils.shapedetector import ShapeDetector


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog,build_detection_test_loader, build_detection_train_loader
from detectron2.projects import point_rend

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

# class CustomTrainerPointrend(DefaultTrainer):

#     @classmethod
#     def build_train_loader(cls, cfg):
#         transform_list = [T.Resize((800, 800)),
#                           T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
#                           T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#                           ]
#         return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=transform_list))
    
def predict(img):
    try:
        DatasetCatalog.remove("coco_train")
        DatasetCatalog.clear()
        MetadataCatalog.clear()
    except BaseException:
        pass
    finally:
        register_coco_instances(f"coco_train", {}, f"Train_coco.json",
                                f"/Images")
        MetadataCatalog.get("coco_train").set(thing_classes=["Corrosion"])
        dataset_dicts = DatasetCatalog.get("coco_train")
        coco_metadata = MetadataCatalog.get("coco_train")

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file('detectron2_repo\projects\PointRend\configs\InstanceSegmentation\pointrend_rcnn_R_50_FPN_3x_coco.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = os.path.join('model_final.pth')
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1

    predictor = DefaultPredictor(cfg)

    outputs = predictor(img)
    print("number of instances")
    print(str(len(outputs["instances"])))
    v = Visualizer(img[:, :, ::-1],
                   metadata=coco_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.SEGMENTATION
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_filename = os.path.join('static', 'img', 'output.jpg')
    cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])
    return output_filename

