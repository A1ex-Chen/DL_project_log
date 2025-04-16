def process_cap(self, img_id, cap):
    if img_id not in self.cap_cache:
        self.cap_cache[img_id] = {}
    if cap not in self.cap_cache[img_id]:
        self.cap_cache[img_id][cap] = {}
        self.cap_cache[img_id][cap]['input_ids'] = self.tokenizer(cap.lower
            (), return_tensors='np')['input_ids'][0, :self.config.
            max_generation_len]
        mention_flag = np.zeros((self.cap_cache[img_id][cap]['input_ids'].
            shape[0], self.obj_cache[img_id]['encoder_input_ids'].shape[0]),
            dtype=np.int64)
        c_input_ids = self.cap_cache[img_id][cap]['input_ids'].tolist()
        en_cls = self.obj_cache[img_id]['encoder_cls'].tolist()
        visit_en_cls = []
        start_pos = {}
        for i, c in enumerate(en_cls):
            if c not in visit_en_cls:
                start_pos[len(visit_en_cls)] = i
                visit_en_cls.append(c)
        start_pos[len(visit_en_cls)] = len(en_cls)
        used_cls = []
        for j, cls_index in enumerate(visit_en_cls):
            if cls_index >= 1601:
                found_word = False
                all_fgs = [fg_index for _, fg_index in self.copy_vocab.
                    d_to_w_group[cls_index]]
                for fg_index in all_fgs:
                    fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
                    s1 = '&'.join([str(f) for f in fg_ch_list])
                    for ch_idx, first_ch in enumerate(c_input_ids):
                        if first_ch == fg_ch_list[0]:
                            s2 = '&'.join([str(f) for f in c_input_ids[
                                ch_idx:ch_idx + len(fg_ch_list)]])
                            if s1 == s2:
                                if ch_idx + len(fg_ch_list) >= len(c_input_ids
                                    ) - 1 or c_input_ids[ch_idx + len(
                                    fg_ch_list)] not in self.attachable_index:
                                    mention_flag[:ch_idx + len(fg_ch_list),
                                        start_pos[j]:start_pos[j + 1]] = 1
                                    if not self.config.static_mf:
                                        mention_flag[ch_idx + len(fg_ch_list):,
                                            start_pos[j]:start_pos[j + 1]] = 2
                                    else:
                                        mention_flag[ch_idx + len(fg_ch_list):,
                                            start_pos[j]:start_pos[j + 1]] = 1
                                    used_cls.append(cls_index)
                                    found_word = True
                                    break
                    if found_word:
                        break
            else:
                mention_flag[:, start_pos[j]:start_pos[j + 1]] = 3
        self.cap_cache[img_id][cap]['mention_flag'] = mention_flag
        self.cap_cache[img_id][cap]['used_cls'] = list(set(used_cls))
