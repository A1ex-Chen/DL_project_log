def synchronize_between_processes(self):
    for iou_type in self.iou_types:
        self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
        create_common_coco_eval(self.coco_eval[iou_type], self.img_ids,
            self.eval_imgs[iou_type])
