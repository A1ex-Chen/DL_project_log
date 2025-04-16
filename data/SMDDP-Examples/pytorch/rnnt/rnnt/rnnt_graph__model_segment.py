def _model_segment(self, encode_block, predict_block, x, x_lens, y, y_lens,
    dict_meta_data=None):
    f, x_lens = encode_block(x, x_lens)
    g = predict_block(y)
    out = self.model.joint(f, g, self.model.apex_transducer_joint, x_lens,
        dict_meta_data)
    return out, x_lens
