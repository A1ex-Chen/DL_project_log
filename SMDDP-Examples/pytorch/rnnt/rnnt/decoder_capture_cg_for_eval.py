def capture_cg_for_eval(self, ema_model, dict_meta_data):
    if type(self.batch_eval_mode
        ) == str and not self.batch_eval_mode.startswith('cg'):
        raise Exception(
            'CUDA graph for eval should only be captured when batch_eval_mode == cg'
            )
    if self.cg_captured == True:
        raise Exception('CUDA graph for eval has been captured previously')
    with torch.no_grad():
        ema_model_eval = self._handle_ema_model(ema_model)
        self.model = ema_model_eval
        feats = torch.ones(dict_meta_data['batch'], dict_meta_data[
            'max_feat_len'], self.rnnt_config['joint_n_hid'], dtype=torch.
            float16, device='cuda')
        feat_lens = torch.ones(dict_meta_data['batch'], dtype=torch.int32,
            device='cuda') * dict_meta_data['max_feat_len']
        self._capture_cg(feats, feat_lens)
