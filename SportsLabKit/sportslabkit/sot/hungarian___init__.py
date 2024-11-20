def __init__(self, target, initial_frame, detection_model=None, image_model
    =None, motion_model=None, matching_fn=None):
    super().__init__(target, pre_init_args={'initial_frame': initial_frame,
        'detection_model': detection_model, 'image_model': image_model,
        'motion_model': motion_model, 'matching_fn': matching_fn})
