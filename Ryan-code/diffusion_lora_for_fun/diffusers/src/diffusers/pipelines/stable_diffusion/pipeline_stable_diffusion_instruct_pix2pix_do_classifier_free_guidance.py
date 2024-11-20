@property
def do_classifier_free_guidance(self):
    return self.guidance_scale > 1.0 and self.image_guidance_scale >= 1.0
