@property
def mask_token_id(self):
    """ Id of the mask token in the vocabulary. E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
    return self.convert_tokens_to_ids(self.mask_token)
