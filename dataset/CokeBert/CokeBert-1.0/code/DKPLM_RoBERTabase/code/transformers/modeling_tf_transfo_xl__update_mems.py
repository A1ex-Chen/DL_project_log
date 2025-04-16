def _update_mems(self, hids, mems, qlen, mlen):
    if mems is None:
        return None
    assert len(hids) == len(mems), 'len(hids) != len(mems)'
    new_mems = []
    end_idx = mlen + max(0, qlen - 0 - self.ext_len)
    beg_idx = max(0, end_idx - self.mem_len)
    for i in range(len(hids)):
        cat = tf.concat([mems[i], hids[i]], axis=0)
        tf.stop_gradient(cat)
        new_mems.append(cat[beg_idx:end_idx])
    return new_mems
