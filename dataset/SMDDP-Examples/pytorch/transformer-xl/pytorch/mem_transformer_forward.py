def forward(self, data, target, mems):
    if mems is None:
        mems = self.init_mems()
    tgt_len = target.size(0)
    hidden, new_mems = self._forward(data, mems=mems)
    pred_hid = hidden[-tgt_len:]
    if self.sample_softmax > 0 and self.training:
        assert self.tie_weight
        logit = sample_logits(self.word_emb, self.out_layer.bias, target,
            pred_hid, self.sampler)
        loss = -F.log_softmax(logit, -1)[:, :, 0]
    else:
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)
    return loss, new_mems
