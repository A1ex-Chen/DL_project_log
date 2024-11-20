@property
def pad_token_id(self):
    """ Id of the padding token in the vocabulary. Log an error if used while not having been set. """
    return self.convert_tokens_to_ids(self.pad_token)
