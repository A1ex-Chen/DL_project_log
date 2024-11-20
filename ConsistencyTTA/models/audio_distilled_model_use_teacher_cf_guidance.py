@property
def use_teacher_cf_guidance(self):
    return (self.teacher_guidance_scale == -1 or self.
        teacher_guidance_scale > 1.0)
