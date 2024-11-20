@classmethod
def from_config(cls, config, custom_objects=None):
    optimizer = tf.keras.optimizers.deserialize(config.pop('optimizer'),
        custom_objects=custom_objects)
    return cls(optimizer, **config)
