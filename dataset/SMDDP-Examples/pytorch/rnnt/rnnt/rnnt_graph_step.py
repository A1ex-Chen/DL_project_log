def step(self, feats, feat_lens, txt, txt_lens, dict_meta_data):
    max_feat_len, encode_block = self.dict_encode_graph[feats.size(0)]
    max_txt_len, predict_block = self.dict_predict_graph[txt.size(1)]
    assert feats.size(0) <= max_feat_len, 'got feat_len of %d' % feats.size(0)
    feats = torch.nn.functional.pad(feats, (0, 0, 0, 0, 0, max_feat_len -
        feats.size(0)))
    txt = torch.nn.functional.pad(txt, (0, max_txt_len - txt.size(1)))
    log_probs, log_prob_lens = self._model_segment(encode_block,
        predict_block, feats, feat_lens, txt, txt_lens, dict_meta_data)
    return log_probs, log_prob_lens
