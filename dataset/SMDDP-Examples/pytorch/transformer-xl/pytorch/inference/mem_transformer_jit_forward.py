def forward(self, data, target, mems: Optional[torch.Tensor]):
    if mems is None:
        mems = self.init_mems()
    tgt_len = target.size(0)
    hidden, new_mems = self._forward(data, mems=mems)
    pred_hid = hidden[-tgt_len:]
    loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
    loss = loss.view(tgt_len, -1)
    return loss, new_mems
