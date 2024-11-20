def get_mention_index(self, cls_index, list_v):
    for _, fg_index in self.copy_vocab.d_to_w_group[cls_index]:
        fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
        s1 = '&'.join([str(f) for f in fg_ch_list])
        for ch_idx, first_ch in enumerate(list_v):
            if first_ch == fg_ch_list[0]:
                s2 = '&'.join([str(f) for f in list_v[ch_idx:ch_idx + len(
                    fg_ch_list)]])
                if s1 == s2:
                    return ch_idx, ch_idx + len(fg_ch_list)
    return -1, -1
