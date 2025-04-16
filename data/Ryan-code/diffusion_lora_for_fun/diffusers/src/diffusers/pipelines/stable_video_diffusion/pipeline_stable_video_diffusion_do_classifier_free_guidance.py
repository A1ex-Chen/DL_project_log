@property
def do_classifier_free_guidance(self):
    if isinstance(self.guidance_scale, (int, float)):
        return self.guidance_scale > 1
    return self.guidance_scale.max() > 1
