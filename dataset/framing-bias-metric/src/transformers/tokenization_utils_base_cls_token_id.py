@cls_token_id.setter
def cls_token_id(self, value):
    self._cls_token = self.convert_tokens_to_ids(value)
