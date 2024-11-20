def cast_and_pad(sample_dict):
    audio = sample_dict['audio']
    label = sample_dict['label']
    paddings = [[0, 16000 - tf.shape(audio)[0]]]
    audio = tf.pad(audio, paddings)
    audio16 = tf.cast(audio, 'int16')
    return audio16, label
