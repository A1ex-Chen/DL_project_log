def forward(self, logits, logit_lens, y, y_lens, dict_meta_data=None):
    if self.precision == 'fp32' and logits.dtype != torch.float32:
        logits = logits.float()
    if y.dtype != torch.int32:
        y = y.int()
    if logit_lens.dtype != torch.int32:
        logit_lens = logit_lens.int()
    if y_lens.dtype != torch.int32:
        y_lens = y_lens.int()
    loss = self.t_loss(logits, y, logit_lens, y_lens, self.blank_idx,
        batch_offset=dict_meta_data['batch_offset'], max_f_len=
        dict_meta_data['max_f_len']).mean()
    return loss
