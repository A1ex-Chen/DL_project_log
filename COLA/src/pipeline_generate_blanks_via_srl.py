def generate_blanks_via_srl(self, text_batch):

    def proper_whitespaces(text):
        return re.sub('\\s([?.!"](?:\\s|$))', '\\1', text.strip())
    srl_list = self.predict_batch_json(text_batch)
    res_list = []
    for srl in srl_list:
        tokens = srl['words']
        verbs = srl['verbs']
        targets = ['ARG0', 'V', 'ARG1']
        blanks = []
        for v in verbs:
            tags = v['tags']
            for tgt in targets:
                if f'B-{tgt}' not in tags:
                    continue
                blk_start = tags.index(f'B-{tgt}')
                blk_end = blk_start + 1 if f'I-{tgt}' not in tags else len(tags
                    ) - tags[::-1].index(f'I-{tgt}')
                sent = ' '.join(tokens[:blk_start]) + ' [BLANK] ' + ' '.join(
                    tokens[blk_end:])
                blanks.append(proper_whitespaces(sent))
        blanks = list(set(blanks))
        if not blanks:
            blanks = ['[BLANK]']
        res_list.append(blanks)
    return res_list
