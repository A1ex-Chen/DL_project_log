@property
def do_classifier_free_guidance(self):
    return (self._guidance_scale > 1 and self.unet.config.
        time_cond_proj_dim is None)
