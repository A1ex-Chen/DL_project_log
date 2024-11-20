@property
def do_classifier_free_guidance(self):
    return self._guidance_scale > 1
