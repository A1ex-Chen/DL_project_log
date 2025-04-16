@classmethod
def from_config(cls, config, custom_objects=None):
    if 'learning_rate' in config:
        if isinstance(config['learning_rate'], dict):
            config['learning_rate'
                ] = tf.keras.optimizers.schedules.deserialize(config[
                'learning_rate'], custom_objects=custom_objects)
    if 'weight_decay' in config:
        if isinstance(config['weight_decay'], dict):
            config['weight_decay'] = tf.keras.optimizers.schedules.deserialize(
                config['weight_decay'], custom_objects=custom_objects)
    return cls(**config)
