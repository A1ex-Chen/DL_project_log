def __call__(self, class_outputs, box_outputs, rpn_box_rois, img_info):
    detection_outputs = dict()
    if self.use_batched_nms:
        generate_detections_fn = generate_detections_gpu
    else:
        generate_detections_fn = generate_detections_tpu
    detections = generate_detections_fn(class_outputs=class_outputs,
        box_outputs=box_outputs, anchor_boxes=rpn_box_rois, image_info=
        img_info, pre_nms_num_detections=self.rpn_post_nms_topn,
        post_nms_num_detections=self.detections_per_image, nms_threshold=
        self.test_nms, class_agnostic_box=self.class_agnostic_box,
        bbox_reg_weights=self.bbox_reg_weights)
    detection_outputs.update({'num_detections': detections[0],
        'detection_boxes': detections[1], 'detection_classes': detections[2
        ], 'detection_scores': detections[3]})
    return detection_outputs
