def get_prompts(self, text, ctrl_codes, blanked_sents, is_complete_blank=True):
    prompts = []
    for tag, bt in itertools.product(ctrl_codes, blanked_sents):
        sep_tok = self.SEP_TOK if bt and is_complete_blank else ''
        prompts.append(
            f'{text.strip()} {self.PERTURB_TOK} [{tag}] {bt.strip()} {sep_tok}'
            .strip())
    return prompts
