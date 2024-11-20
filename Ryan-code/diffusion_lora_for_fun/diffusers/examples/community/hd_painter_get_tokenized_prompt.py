def get_tokenized_prompt(self, prompt):
    out = self.tokenizer(prompt)
    return [self.tokenizer.decode(x) for x in out['input_ids']]
