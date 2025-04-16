@property
def all_special_ids(self):
    """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
    all_toks = self.all_special_tokens
    all_ids = self.convert_tokens_to_ids(all_toks)
    return all_ids
