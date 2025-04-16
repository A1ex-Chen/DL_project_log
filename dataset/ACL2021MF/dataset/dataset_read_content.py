def read_content(self, json_path):
    print('reading data from %s ...' % json_path)
    self.record = []
    with open(json_path) as out:
        lines = out.readlines()
        for l in tqdm(lines):
            item = json.loads(l.strip())
            concept_set = ' '.join(item['concept_set'].split('#'))
            concept_set = 'generate a sentence with these concepts : '
            concept_set_input_ids = self.tokenizer(concept_set,
                return_tensors='np')['input_ids'][0, :-1].tolist()
            concept_cls = []
            for concept in item['concept_set'].split('#'):
                if concept in self.copy_vocab.word_to_category_id:
                    concept_cls.append(self.copy_vocab.word_to_category_id[
                        concept])
                else:
                    fg_index = self.copy_vocab.w_to_i[concept]
                    concept_cls.append(self.copy_vocab.i_to_cls[fg_index])
            assert len(concept_cls) <= 5
            start_pos = []
            for i, c_cls in enumerate(concept_cls):
                start_pos.append(len(concept_set_input_ids))
                concept_set_input_ids += self.copy_vocab.token_class[c_cls]
                if i == len(concept_cls) - 1:
                    concept_set_input_ids.append(self.tokenizer.eos_token_id)
            start_pos.append(len(concept_set_input_ids) - 1)
            position_indicator = np.zeros((5, len(concept_set_input_ids)),
                dtype=np.float32)
            for i in range(len(concept_cls)):
                position_indicator[i, start_pos[i]:start_pos[i + 1]] = 1
            sum_check = np.sum(position_indicator, axis=1)
            for i in range(len(concept_cls)):
                assert sum_check[i] > 0
            cls_on_input = np.zeros((len(concept_set_input_ids),), dtype=np
                .int64)
            for i, cls_ in enumerate(concept_cls):
                cls_on_input[start_pos[i]:start_pos[i + 1]] = cls_
            gt = copy.deepcopy(item['scene'])
            if self.is_training:
                for c in item['scene']:
                    c = c.lower()
                    c_input_ids = self.tokenizer(c, return_tensors='np')[
                        'input_ids'][0]
                    string_caption = ' '.join([str(x) for x in c_input_ids])
                    if self.config.use_pointer:
                        for c_cls in concept_cls:
                            for _, fg_index in self.copy_vocab.d_to_w_group[
                                c_cls]:
                                fg_word_index = self.copy_vocab.token_fg_w[
                                    fg_index]
                                fg_str = ' '.join([str(x) for x in
                                    fg_word_index])
                                fg_softmax_index = (self.config.vocab_size +
                                    fg_index)
                                string_caption = re.sub(' %s ' % fg_str, 
                                    ' (%d) ' % fg_softmax_index, string_caption
                                    )
                                string_caption = re.sub(' %s$' % fg_str, 
                                    ' (%d)' % fg_softmax_index, string_caption)
                                string_caption = re.sub('^%s ' % fg_str, 
                                    '(%d) ' % fg_softmax_index, string_caption)
                    c_input_ids = []
                    str_id_list = string_caption.split()
                    copy_mention_flag = np.zeros((len(str_id_list) + 1, 5))
                    decoder_mention_flag = np.zeros((len(str_id_list), len(
                        concept_set_input_ids)))
                    use_this_record = True
                    if self.config.use_pointer:
                        for index, w in enumerate(str_id_list):
                            if w.startswith('(') and w.endswith(')'):
                                fg_index = int(w[1:-1])
                                cls_index = self.copy_vocab.i_to_cls[
                                    fg_index - self.config.vocab_size]
                                for j in range(len(concept_cls)):
                                    if concept_cls[j] == cls_index:
                                        copy_mention_flag[:index + 1, j] = 1
                                        copy_mention_flag[index + 1:, j] = 2
                                        decoder_mention_flag[:index + 1,
                                            start_pos[j]:start_pos[j + 1]] = 1
                                        decoder_mention_flag[index + 1:,
                                            start_pos[j]:start_pos[j + 1]] = 2
                                c_input_ids.append(fg_index)
                                assert c_input_ids[-1] > self.config.vocab_size
                            else:
                                c_input_ids.append(int(w))
                                assert c_input_ids[-1
                                    ] <= self.config.vocab_size
                    else:
                        for index, w in enumerate(str_id_list):
                            c_input_ids.append(int(w))
                            assert c_input_ids[-1] <= self.config.vocab_size
                        for j, cls_index in enumerate(concept_cls):
                            if cls_index == 0:
                                continue
                            for _, fg_index in self.copy_vocab.d_to_w_group[
                                cls_index]:
                                fg_ch_list = self.copy_vocab.token_fg_w[
                                    fg_index]
                                s1 = '&'.join([str(f) for f in fg_ch_list])
                                for ch_idx, first_ch in enumerate(c_input_ids):
                                    if first_ch == fg_ch_list[0]:
                                        s2 = '&'.join([str(f) for f in
                                            c_input_ids[ch_idx:ch_idx + len(
                                            fg_ch_list)]])
                                        if s1 == s2:
                                            if ch_idx + len(fg_ch_list) >= len(
                                                c_input_ids) - 1 or c_input_ids[
                                                ch_idx + len(fg_ch_list)
                                                ] not in self.attachable_index:
                                                decoder_mention_flag[:ch_idx + len(
                                                    fg_ch_list), start_pos[j]:start_pos
                                                    [j + 1]] = 1
                                                if not self.config.static_mf:
                                                    decoder_mention_flag[ch_idx + len(
                                                        fg_ch_list):, start_pos[j]:
                                                        start_pos[j + 1]] = 2
                                                else:
                                                    decoder_mention_flag[ch_idx + len(
                                                        fg_ch_list):, start_pos[j]:
                                                        start_pos[j + 1]] = 1
                                                break
                        if not self.config.static_mf:
                            for j in range(len(concept_cls)):
                                for jj in range(start_pos[j], start_pos[j + 1]
                                    ):
                                    if not decoder_mention_flag[-1, jj] == 2:
                                        use_this_record = False
                    if use_this_record:
                        instance_tuple = (concept_set_input_ids,
                            position_indicator, cls_on_input, concept_cls,
                            copy_mention_flag, decoder_mention_flag,
                            c_input_ids, gt, item['concept_set'].split('#'))
                        self.record.append(instance_tuple)
            else:
                copy_mention_flag = np.zeros((1, 5))
                copy_mention_flag[0, :len(concept_cls)] = 1
                decoder_mention_flag = np.zeros((1, len(concept_set_input_ids))
                    )
                decoder_mention_flag[0, start_pos[0]:start_pos[-1]] = 1
                self.record.append((concept_set_input_ids,
                    position_indicator, cls_on_input, concept_cls,
                    copy_mention_flag, decoder_mention_flag, None, gt, item
                    ['concept_set'].split('#')))
    if self.is_training:
        random.shuffle(self.record)
