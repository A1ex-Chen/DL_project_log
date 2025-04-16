@property
def dummy_inputs(self):
    inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 
        4, 5]])
    attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1,
        1]])
    if self.config.use_lang_emb and self.config.n_langs > 1:
        langs_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 
            0, 1, 1]])
    else:
        langs_list = None
    return [inputs_list, attns_list, langs_list]
