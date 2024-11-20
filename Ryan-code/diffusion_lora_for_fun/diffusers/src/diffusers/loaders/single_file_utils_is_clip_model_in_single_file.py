def is_clip_model_in_single_file(class_obj, checkpoint):
    is_clip_in_checkpoint = any([is_clip_model(checkpoint),
        is_open_clip_model(checkpoint), is_open_clip_sdxl_model(checkpoint),
        is_open_clip_sdxl_refiner_model(checkpoint)])
    if (class_obj.__name__ == 'CLIPTextModel' or class_obj.__name__ ==
        'CLIPTextModelWithProjection') and is_clip_in_checkpoint:
        return True
    return False
