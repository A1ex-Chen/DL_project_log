def _update_mems(self, hids, mems, qlen, mlen):
    if mems is None:
        return None
    assert len(hids) == len(mems), 'len(hids) != len(mems)'
    with torch.no_grad():
        stacked = torch.stack(hids)
        if self.mem_len == self.tgt_len and self.ext_len == 0 and stacked.size(
            1) == self.mem_len:
            new_mems = stacked.detach()
        else:
            end_idx = mlen + max(0, qlen - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            if mems.numel():
                cat = torch.cat([mems, stacked], dim=1)
            else:
                cat = stacked
            new_mems = cat[:, beg_idx:end_idx].detach()
    return new_mems
