@eos_token_id.setter
def eos_token_id(self, value):
    self._eos_token = self.convert_tokens_to_ids(value)
