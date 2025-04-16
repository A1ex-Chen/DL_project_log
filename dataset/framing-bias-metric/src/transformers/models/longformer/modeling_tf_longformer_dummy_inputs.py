@property
def dummy_inputs(self):
    input_ids = tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)
    global_attention_mask = tf.constant([[[0, 0, 0, 1], [0, 0, 0, 1]]] * 2)
    return {'input_ids': input_ids, 'global_attention_mask':
        global_attention_mask}
