def _update_mems(self, hids, mems, qlen, mlen):
    if mems is None:
        return None
    assert len(hids) == len(mems), 'len(hids) != len(mems)'
    with torch.no_grad():
        new_mems = []
        end_idx = mlen + max(0, qlen - 0 - self.ext_len)
        beg_idx = max(0, end_idx - self.mem_len)
        for i in range(len(hids)):
            cat = torch.cat([mems[i], hids[i]], dim=0)
            new_mems.append(cat[beg_idx:end_idx].detach())
    return new_mems
