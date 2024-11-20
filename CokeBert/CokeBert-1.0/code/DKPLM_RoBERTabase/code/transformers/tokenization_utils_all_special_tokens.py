@property
def all_special_tokens(self):
    """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
    all_toks = []
    set_attr = self.special_tokens_map
    for attr_value in set_attr.values():
        all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (
            list, tuple)) else [attr_value])
    all_toks = list(set(all_toks))
    return all_toks
