from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou




class COCOEvaluatorFPN(COCOEvaluator):
