def __init__(self, use_batched_nms=False, rpn_post_nms_topn=1000,
    detections_per_image=100, test_nms=0.5, class_agnostic_box=False,
    bbox_reg_weights=(10.0, 10.0, 5.0, 5.0)):
    self.use_batched_nms = use_batched_nms
    self.rpn_post_nms_topn = rpn_post_nms_topn
    self.detections_per_image = detections_per_image
    self.test_nms = test_nms
    self.bbox_reg_weights = bbox_reg_weights
    self.class_agnostic_box = class_agnostic_box
