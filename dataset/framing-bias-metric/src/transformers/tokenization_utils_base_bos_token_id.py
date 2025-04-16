@bos_token_id.setter
def bos_token_id(self, value):
    self._bos_token = self.convert_tokens_to_ids(value)
