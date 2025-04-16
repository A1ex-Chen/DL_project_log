def _capture_cg_for_main_loop(self, list_input_tensor):
    if self.batch_eval_mode == 'cg_unroll_pipeline':
        func_to_be_captured = self._eval_main_loop_unroll
    else:
        func_to_be_captured = self._eval_main_loop_stream
    self.label_upd_stream = torch.cuda.Stream()
    self.hidden_upd_stream = torch.cuda.Stream()
    self.time_idx_upd_stream = torch.cuda.Stream()
    cg = graph_simple(func_to_be_captured, tuple(t.clone() for t in
        list_input_tensor), torch.cuda.Stream(), warmup_iters=2)
    return cg
