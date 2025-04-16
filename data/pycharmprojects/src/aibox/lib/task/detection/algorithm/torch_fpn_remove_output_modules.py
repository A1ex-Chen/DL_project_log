def remove_output_modules(self):
    del self.net.rpn.head.cls_logits
    del self.net.rpn.head.bbox_pred
    del self.net.roi_heads.box_predictor.cls_score
    del self.net.roi_heads.box_predictor.bbox_pred
