def warmup_lr():
    return self._rescaled_lr * (step / tf.cast(self._warmup_steps, tf.float32))
