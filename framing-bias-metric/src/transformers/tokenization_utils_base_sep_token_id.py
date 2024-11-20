@sep_token_id.setter
def sep_token_id(self, value):
    self._sep_token = self.convert_tokens_to_ids(value)
