@property
def sep_token_id(self):
    """ Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
    return self.convert_tokens_to_ids(self.sep_token)
