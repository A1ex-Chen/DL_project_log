def initialize(self):
    """Initializes backend related initializations."""
    if tf.config.list_physical_devices('GPU'):
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'
    tf.keras.backend.set_image_data_format(data_format)
    if self.params.run_eagerly:
        tf.config.experimental_run_functions_eagerly(True)
