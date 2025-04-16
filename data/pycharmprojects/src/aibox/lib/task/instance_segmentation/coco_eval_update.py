def update(self, predictions):
    img_ids = list(np.unique(list(predictions.keys())))
    self.img_ids.extend(img_ids)
    for iou_type in self.iou_types:
        results = self.prepare(predictions, iou_type)
        coco_dt = loadRes(self.coco_gt, results) if results else COCO()
        coco_eval = self.coco_eval[iou_type]
        coco_eval.cocoDt = coco_dt
        coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(coco_eval)
        self.eval_imgs[iou_type].append(eval_imgs)
