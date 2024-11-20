def convert_to_int16(sample_dict):
    audio = sample_dict['audio']
    label = sample_dict['label']
    audio16 = tf.cast(audio, 'int16')
    return audio16, label
