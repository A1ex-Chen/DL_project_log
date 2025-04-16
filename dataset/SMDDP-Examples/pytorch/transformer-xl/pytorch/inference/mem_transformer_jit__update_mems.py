def _update_mems(self, hids: List[torch.Tensor], mems: torch.Tensor, qlen:
    int, mlen: int):
    assert len(hids) == len(mems), 'len(hids) != len(mems)'
    stacked = torch.stack(hids)
    end_idx = mlen + max(0, qlen - self.ext_len)
    beg_idx = max(0, end_idx - self.mem_len)
    if mems.numel():
        cat = torch.cat([mems, stacked], dim=1)
    else:
        cat = stacked
    new_mems = cat[:, beg_idx:end_idx].detach()
    return new_mems
