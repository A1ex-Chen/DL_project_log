def remove_output_modules(self):
    del self.rpn_head.anchor_objectness
    del self.rpn_head.anchor_transformer
    del self.roi_head.proposal_class
    del self.roi_head.proposal_transformer
