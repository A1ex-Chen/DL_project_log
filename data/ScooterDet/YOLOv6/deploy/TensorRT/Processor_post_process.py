def post_process(self, outputs, img_shape, conf_thres=0.5, iou_thres=0.6):
    if self.is_end2end:
        det_t = outputs
    else:
        det_t = self.non_max_suppression(outputs, conf_thres, iou_thres,
            multi_label=True)
    self.scale_coords(self.input_shape, det_t[0][:, :4], img_shape[0],
        img_shape[1])
    return det_t[0]
