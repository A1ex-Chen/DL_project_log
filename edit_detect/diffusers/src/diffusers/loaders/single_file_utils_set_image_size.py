def set_image_size(checkpoint, image_size=None):
    if image_size:
        return image_size
    model_type = infer_diffusers_model_type(checkpoint)
    image_size = DIFFUSERS_TO_LDM_DEFAULT_IMAGE_SIZE_MAP[model_type]
    return image_size
