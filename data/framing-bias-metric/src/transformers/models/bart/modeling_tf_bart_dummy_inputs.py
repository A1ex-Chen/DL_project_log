@property
def dummy_inputs(self):
    pad_token = 1
    input_ids = tf.cast(tf.constant(DUMMY_INPUTS), tf.int32)
    decoder_input_ids = tf.cast(tf.constant(DUMMY_INPUTS), tf.int32)
    dummy_inputs = {'decoder_input_ids': decoder_input_ids,
        'attention_mask': tf.math.not_equal(input_ids, pad_token),
        'input_ids': input_ids}
    return dummy_inputs
