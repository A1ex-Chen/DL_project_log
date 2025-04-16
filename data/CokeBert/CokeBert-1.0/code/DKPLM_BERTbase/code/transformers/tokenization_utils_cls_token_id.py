@property
def cls_token_id(self):
    """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
    return self.convert_tokens_to_ids(self.cls_token)
