@tf.function(experimental_relax_shapes=True)
def __call__(self, step):
    global_step_recomp = tf.cast(step, self.dtype)
    if global_step_recomp >= self.warmup_steps:
        if self.overlap:
            return self.scheduler(global_step_recomp)
        return self.scheduler(global_step_recomp - self.warmup_steps)
    return self.compute_linear_warmup(global_step_recomp)
