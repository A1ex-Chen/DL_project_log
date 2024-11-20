def __init__(self, bbox_head, bbox_roi_extractor, bbox_sampler, box_encoder,
    inference_detector, stage_weights, mask_head=None, mask_roi_extractor=
    None, name='CascadeRoIHead', trainable=True, *args, **kwargs):
    super(CascadeRoIHead, self).__init__(*args, bbox_head=bbox_head,
        bbox_roi_extractor=bbox_roi_extractor, bbox_sampler=bbox_sampler,
        box_encoder=box_encoder, inference_detector=inference_detector,
        mask_head=mask_head, mask_roi_extractor=mask_roi_extractor, name=
        name, trainable=trainable, **kwargs)
    self.stage_weights = [(i / sum(stage_weights)) for i in stage_weights]
    self.num_stages = len(self.bbox_head)
