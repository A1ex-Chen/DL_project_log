def read_content(self, json_path):
    print('reading data from %s ...' % json_path)
    self.record = []
    with open(json_path) as out:
        instances = json.loads(out.read())
    total_input = 0
    match_input = 0
    type_info_ratio = {}
    for ins_id, instance in tqdm(enumerate(instances)):
        gt = copy.deepcopy(instance['ref'])
        gt_mr = []
        new_mr_input = []
        for mr in instance['mr']:
            each_mr = mr.replace('[', ' ')
            each_mr = each_mr.replace(']', ' ')
            each_mr = each_mr.strip()
            words = each_mr.split()
            if words[0] in self.keyword_norm:
                tag = self.keyword_norm[words[0]]
                value = ' '.join([x.strip() for x in words[1:]])
            else:
                tag = words[0] if not words[0] == 'customer' else ' '.join(
                    words[:2])
                value = ' '.join([x.strip() for x in words[1:]]) if not words[0
                    ] == 'customer' else ' '.join([x.strip() for x in words
                    [2:]])
            new_mr_input.append(tag + ' ' + value)
            gt_mr.append((mr, value))
        input_cls_info = []
        for text, gt_m in zip(new_mr_input, gt_mr):
            cls_id = self.copy_vocab.word_to_category_id[gt_m[0]]
            m_input = self.tokenizer(text, return_tensors='np')['input_ids'][
                0, :-1].tolist()
            input_cls_info.append((cls_id, m_input, gt_m[1]))
        encoder_input = []
        encoder_cls = []
        for cls_id, m_input, _ in input_cls_info:
            encoder_input += m_input
            encoder_cls += [cls_id] * len(m_input)
        encoder_input.append(self.tokenizer.eos_token_id)
        encoder_cls.append(0)
        encoder_input = np.array(encoder_input, dtype=np.int64)
        encoder_cls = np.array(encoder_cls, dtype=np.int64)
        if not self.is_training:
            mention_flag = np.array(encoder_cls > 0, dtype=np.int64)
            mention_flag = mention_flag[np.newaxis, :]
            self.record.append((ins_id, encoder_input, encoder_cls,
                mention_flag, None, gt, gt_mr))
        else:
            for v in instance['ref']:
                ref = v
                v = ' '.join(v.split()).lower()
                v = self.tokenizer(v, return_tensors='np')['input_ids'][0,
                    :self.config.max_generation_len]
                list_v = v.tolist()
                mentioned_cls_pos = []
                for cls_id, m_input, name in input_cls_info:
                    s_pos, e_pos = self.get_mention_index(cls_id, list_v)
                    mentioned_cls_pos.append((s_pos, e_pos, cls_id))
                    total_input += 1
                    if s_pos >= 0 and e_pos >= 0:
                        match_input += 1
                encoder_input = np.array(encoder_input, dtype=np.int64)
                encoder_cls = np.array(encoder_cls, dtype=np.int64)
                mention_flag = np.zeros((v.shape[0], encoder_input.shape[0]
                    ), dtype=np.int64)
                for s_pos, e_pos, cls_id in mentioned_cls_pos:
                    for e_index in range(encoder_cls.shape[0]):
                        if encoder_cls[e_index] == cls_id:
                            if e_pos >= 0:
                                mention_flag[:e_pos, e_index] = 1
                                if not self.config.static_mf:
                                    mention_flag[e_pos:, e_index] = 2
                                else:
                                    mention_flag[e_pos:, e_index] = 1
                            else:
                                mention_flag[:, e_index
                                    ] = 0 if not self.config.use_mf_merged else 1
                self.record.append((ins_id, encoder_input, encoder_cls,
                    mention_flag, v, gt, gt_mr))
    if self.is_training:
        random.shuffle(self.record)
        print('Match Ratio %.2f' % (100 * match_input / total_input))
