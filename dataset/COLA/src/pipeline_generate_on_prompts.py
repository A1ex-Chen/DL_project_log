def generate_on_prompts(self, generator, prompts, **kwargs):

    def split_ctrl_code(text):
        r = re.search('\\[(?P<code>[a-z]+)\\](?P<text>.+)', text)
        if r:
            return r.group('code').strip(), r.group('text').strip()
        return '', text

    def remove_blanks(text):
        try:
            before, answers = text.split(self.SEP_TOK)
            answers = [x.strip() for x in answers.split(self.ANSWER_TOK)][:-1]
            answers = [(x if x != self.EMPTY_TOK else '') for x in answers]
            for a in answers:
                if a == '':
                    before = re.sub(' %s' % re.escape(self.BLANK_TOK), a,
                        before, count=1)
                else:
                    before = re.sub('%s' % re.escape(self.BLANK_TOK), a,
                        before, count=1)
            return before, answers
        except:
            return text, []

    def batched_generate(generator, examples, **kwargs):
        preds_list = []
        with torch.no_grad():
            for e in range(len(examples)):
                preds_list += generator(examples[e:e + 1], early_stopping=
                    True, **kwargs)
        return preds_list
    preds_list = batched_generate(generator, prompts, **kwargs)
    if len(prompts) == 1:
        preds_list = [preds_list]
    preds_list_cleaned = []
    for prompt, preds in zip(prompts, preds_list):
        prev_list = set()
        for s in preds:
            total_sequence = s['generated_text'].split(self.PERTURB_TOK)[-1]
            normalized, _ = remove_blanks(total_sequence)
            input_ctrl_code, normalized = split_ctrl_code(normalized)
            prev_list.add((input_ctrl_code, normalized))
        preds_list_cleaned.append(list(prev_list))
    return preds_list_cleaned
