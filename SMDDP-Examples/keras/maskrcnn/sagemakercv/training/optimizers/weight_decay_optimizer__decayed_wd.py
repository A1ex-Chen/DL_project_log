def _decayed_wd(self, var_dtype):
    wd_t = self._get_hyper('weight_decay', var_dtype)
    if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
        wd_t = tf.cast(wd_t(self.iterations), var_dtype)
    return wd_t
