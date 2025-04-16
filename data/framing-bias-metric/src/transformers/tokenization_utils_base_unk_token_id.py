@unk_token_id.setter
def unk_token_id(self, value):
    self._unk_token = self.convert_tokens_to_ids(value)
