def init_infer_state(self, batch_size, total_token_num, max_input_len):
    self.infer_state = dict(is_prefill=True, batch_size=batch_size,
        total_token_num=total_token_num, max_len_in_batch=max_input_len)
