@property
def eos_token_id(self):
    """ Id of the end of sentence token in the vocabulary. Log an error if used while not having been set. """
    return self.convert_tokens_to_ids(self.eos_token)
