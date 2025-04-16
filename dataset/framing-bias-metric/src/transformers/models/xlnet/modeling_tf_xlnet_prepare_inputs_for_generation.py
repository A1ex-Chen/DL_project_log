def prepare_inputs_for_generation(self, inputs, past, use_mems=None, **kwargs):
    offset = 2
    effective_batch_size = inputs.shape[0]
    dummy_token = tf.zeros((effective_batch_size, 1), dtype=tf.int32)
    if past:
        inputs = tf.concat([inputs[:, -offset:], dummy_token], axis=1)
    else:
        inputs = tf.concat([inputs, dummy_token], axis=1)
    sequence_length = inputs.shape[1]
    perm_mask = tf.zeros((effective_batch_size, sequence_length, 
        sequence_length - 1), dtype=tf.float32)
    perm_mask_seq_end = tf.ones((effective_batch_size, sequence_length, 1),
        dtype=tf.float32)
    perm_mask = tf.concat([perm_mask, perm_mask_seq_end], axis=-1)
    target_mapping = tf.zeros((effective_batch_size, 1, sequence_length - 1
        ), dtype=tf.float32)
    target_mapping_seq_end = tf.ones((effective_batch_size, 1, 1), dtype=tf
        .float32)
    target_mapping = tf.concat([target_mapping, target_mapping_seq_end],
        axis=-1)
    inputs = {'input_ids': inputs, 'perm_mask': perm_mask, 'target_mapping':
        target_mapping, 'use_mems': kwargs.get('use_mems')}
    if past:
        inputs['mems'] = tuple(layer_past[:-offset, :, :] for layer_past in
            past)
    return inputs
