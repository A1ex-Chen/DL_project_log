def text2emb(self, text, add_special=False):
    to_regress_tokens = self.tokenizer(text, return_tensors='pt', padding=
        'longest', truncation=True, add_special_tokens=add_special).to(self
        .device)
    targets = self.mask_human_targets(to_regress_tokens.input_ids)
    targets = targets.to(self.device)
    return to_regress_tokens, targets
