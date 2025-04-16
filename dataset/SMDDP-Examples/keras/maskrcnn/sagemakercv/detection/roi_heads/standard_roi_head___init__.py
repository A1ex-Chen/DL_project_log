def __init__(self, bbox_head, bbox_roi_extractor, bbox_sampler, box_encoder,
    inference_detector, mask_head=None, mask_roi_extractor=None, name=
    'StandardRoIHead', trainable=True, *args, **kwargs):
    super(StandardRoIHead, self).__init__(*args, name=name, trainable=
        trainable, **kwargs)
    self.bbox_head = bbox_head
    self.bbox_roi_extractor = bbox_roi_extractor
    self.bbox_sampler = bbox_sampler
    self.box_encoder = box_encoder
    self.mask_head = mask_head
    self.mask_roi_extractor = (mask_roi_extractor if mask_roi_extractor is not
        None else self.bbox_roi_extractor)
    self.inference_detector = inference_detector
