def summarize(self):
    for iou_type, coco_eval in self.coco_eval.items():
        print('IoU metric: {}'.format(iou_type))
        coco_eval.summarize()
