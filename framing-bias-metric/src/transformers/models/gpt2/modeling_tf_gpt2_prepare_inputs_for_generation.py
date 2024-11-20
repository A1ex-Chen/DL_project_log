def prepare_inputs_for_generation(self, inputs, past, **kwargs):
    if past:
        inputs = tf.expand_dims(inputs[:, -1], -1)
    return {'input_ids': inputs, 'past': past, 'use_cache': kwargs['use_cache']
        }
