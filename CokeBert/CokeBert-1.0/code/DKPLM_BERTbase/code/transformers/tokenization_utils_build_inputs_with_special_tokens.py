def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
    logger.warning(
        'This tokenizer does not make use of special tokens. Input is returned with no modification.'
        )
    if token_ids_1 is None:
        return token_ids_0
    return token_ids_0 + token_ids_1
