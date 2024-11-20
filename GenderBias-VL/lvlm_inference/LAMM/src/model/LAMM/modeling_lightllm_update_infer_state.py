def update_infer_state(self):
    self.infer_state['total_token_num'] += self.infer_state['batch_size']
    self.infer_state['max_len_in_batch'] += 1
    self.infer_state['is_prefill'] = False
