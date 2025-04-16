def _gen_encode_graph(self, max_feat_len):
    feats = torch.ones(max_feat_len, self.batch_size, self.rnnt_config[
        'in_feats'], dtype=torch.float16, device='cuda')
    feat_lens = torch.ones(self.batch_size, dtype=torch.int32, device='cuda'
        ) * max_feat_len
    encode_args = feats, feat_lens
    rnnt_encode = RNNTEncode(self.model.encoder, self.model.joint_enc, self
        .model.min_lstm_bs)
    encode_segment = graph(rnnt_encode, encode_args, self.cg_stream,
        warmup_iters=2, warmup_only=False)
    return encode_segment
