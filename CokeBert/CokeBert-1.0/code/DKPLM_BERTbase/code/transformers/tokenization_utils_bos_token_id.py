@property
def bos_token_id(self):
    """ Id of the beginning of sentence token in the vocabulary. Log an error if used while not having been set. """
    return self.convert_tokens_to_ids(self.bos_token)
