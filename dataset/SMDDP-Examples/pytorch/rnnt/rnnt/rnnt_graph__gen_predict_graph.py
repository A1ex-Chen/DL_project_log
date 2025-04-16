def _gen_predict_graph(self, max_txt_len):
    txt = torch.ones(self.batch_size, max_txt_len, dtype=torch.int64,
        device='cuda')
    predict_args = txt,
    rnnt_predict = RNNTPredict(self.model.prediction, self.model.joint_pred,
        self.model.min_lstm_bs)
    predict_segment = graph(rnnt_predict, predict_args, self.cg_stream,
        warmup_iters=2, warmup_only=False)
    return predict_segment
