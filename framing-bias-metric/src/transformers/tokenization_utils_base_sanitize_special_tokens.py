def sanitize_special_tokens(self) ->int:
    """
        Make sure that all the special tokens attributes of the tokenizer (:obj:`tokenizer.mask_token`,
        :obj:`tokenizer.cls_token`, etc.) are in the vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            :obj:`int`: The number of tokens added in the vocabulary during the operation.
        """
    return self.add_tokens(self.all_special_tokens_extended, special_tokens
        =True)
