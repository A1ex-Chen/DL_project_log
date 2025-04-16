@property
def dummy_inputs(self):
    inputs = tf.constant(DUMMY_INPUTS)
    input_mask = tf.constant(DUMMY_MASK)
    dummy_inputs = {'input_ids': inputs, 'decoder_input_ids': inputs,
        'decoder_attention_mask': input_mask}
    return dummy_inputs
