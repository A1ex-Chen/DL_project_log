@pad_token_id.setter
def pad_token_id(self, value):
    self._pad_token = self.convert_tokens_to_ids(value)
