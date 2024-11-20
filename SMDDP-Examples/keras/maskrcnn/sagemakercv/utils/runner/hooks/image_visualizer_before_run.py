@master_only
def before_run(self, runner):
    self.inference_detector = BoxDetector(use_batched_nms=runner.cfg.MODEL.
        INFERENCE.USE_BATCHED_NMS, rpn_post_nms_topn=runner.cfg.MODEL.
        INFERENCE.POST_NMS_TOPN, detections_per_image=runner.cfg.MODEL.
        INFERENCE.DETECTIONS_PER_IMAGE, test_nms=runner.cfg.MODEL.INFERENCE
        .DETECTOR_NMS, class_agnostic_box=runner.cfg.MODEL.INFERENCE.
        CLASS_AGNOSTIC, bbox_reg_weights=runner.cfg.MODEL.BBOX_REG_WEIGHTS)
    self.thread_pool = ThreadPoolExecutor(max_workers=4)
