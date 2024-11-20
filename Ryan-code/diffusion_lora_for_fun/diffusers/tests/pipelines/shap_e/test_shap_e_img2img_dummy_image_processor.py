@property
def dummy_image_processor(self):
    image_processor = CLIPImageProcessor(crop_size=224, do_center_crop=True,
        do_normalize=True, do_resize=True, image_mean=[0.48145466, 
        0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 
        0.27577711], resample=3, size=224)
    return image_processor
