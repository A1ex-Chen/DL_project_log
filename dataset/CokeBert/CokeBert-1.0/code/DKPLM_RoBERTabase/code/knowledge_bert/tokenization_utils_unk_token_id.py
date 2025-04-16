@property
def unk_token_id(self):
    """ Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
    return self.convert_tokens_to_ids(self.unk_token)
