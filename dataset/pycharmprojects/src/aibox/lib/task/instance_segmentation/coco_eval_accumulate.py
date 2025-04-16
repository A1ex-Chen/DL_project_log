def accumulate(self):
    for coco_eval in self.coco_eval.values():
        coco_eval.accumulate()
