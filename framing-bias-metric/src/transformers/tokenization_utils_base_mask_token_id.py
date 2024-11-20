@mask_token_id.setter
def mask_token_id(self, value):
    self._mask_token = self.convert_tokens_to_ids(value)
