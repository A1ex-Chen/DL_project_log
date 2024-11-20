def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
    """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
    sep = [self.sep_token_id]
    cls = [self.cls_token_id]
    if token_ids_1 is None:
        return len(cls + token_ids_0 + sep) * [0]
    return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
