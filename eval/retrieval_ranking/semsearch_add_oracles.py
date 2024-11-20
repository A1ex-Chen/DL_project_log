def add_oracles(self, set_oracle, gt_sentence, gt_sent_idx):
    if not self.contextual:
        self.list_oracle = list(set_oracle)
        return self.list_oracle
    else:
        self.list_oracle.clear()
        if gt_sent_idx == -1:
            return []
        answer_tokens = [token.text.lower() for token in self.nlp.tokenizer
            (list(set_oracle)[0])]
        if answer_tokens == ['u.k', '.', 'branch']:
            answer_tokens = ['u.k.', 'branch']
        elif answer_tokens == ['u.s', '.', 'government', 'statistics']:
            answer_tokens = ['u.s.', 'government', 'statistics']
        elif answer_tokens == ['u.s', '.', 'border']:
            answer_tokens = ['u.s.', 'border']
        elif answer_tokens == ['u.s', '.', 'government', 'control']:
            answer_tokens = ['u.s.', 'government', 'control']
        elif answer_tokens == ['u.s', '.', 'mail']:
            answer_tokens = ['u.s.', 'mail']
        for phrase, index in self.extractor.extract_with_index(gt_sentence,
            ngram_min=2, ngram_max=len(answer_tokens)):
            if ' '.join(answer_tokens) in phrase.strip():
                self.list_oracle.append((' '.join(answer_tokens), index,
                    gt_sent_idx))
        if len(self.list_oracle) == 0:
            print('FAILED TO ADD ORACLE {} WITH TEXT {}'.format(list(
                set_oracle), gt_sentence))
        return [' '.join(answer_tokens)]
