@property
@tf_required
def gpu_list(self):
    return tf.config.list_physical_devices('GPU')
