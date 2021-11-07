import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"),
)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 600
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)


def analyse(image, use_max_area=0):
    outputs = predictor(image)
    predictions = outputs['instances']
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    areas = []
    max_damage_area = float('-inf')
    total_damage_area = 0
    max_idx = None

    for i in range(len(boxes)):
        area = boxes[i].area()
        if area > max_damage_area:
            max_idx = i
            max_damage_area = area
        total_damage_area += area
        areas.append(area)

    # cv2.rectangle(
    #     image,
    #     tuple(boxes[max_idx].tensor[0][0:2].int().numpy()),
    #     tuple(boxes[max_idx].tensor[0][2:].int().numpy()),
    #     (0, 255, 0),
    #     2,
    # )

    if use_max_area is True:
        damage_area = max_damage_area
    else:
        damage_area = total_damage_area

    image_area = image.shape[0] * image.shape[1]
    damage_area = (torch.div(damage_area, image_area) * 100).item()

    return damage_area > 5, {
        'damage': f'{damage_area}%'
    }
