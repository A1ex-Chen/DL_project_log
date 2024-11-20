def __init__(self, coco_gt, iou_types):
    assert isinstance(iou_types, (list, tuple))
    coco_gt = copy.deepcopy(coco_gt)
    self.coco_gt = coco_gt
    self.iou_types = iou_types
    self.coco_eval = {}
    for iou_type in iou_types:
        self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
    self.img_ids = []
    self.eval_imgs = {k: [] for k in iou_types}
