@additional_special_tokens_ids.setter
def additional_special_tokens_ids(self, values):
    self._additional_special_tokens = [self.convert_tokens_to_ids(value) for
        value in values]
