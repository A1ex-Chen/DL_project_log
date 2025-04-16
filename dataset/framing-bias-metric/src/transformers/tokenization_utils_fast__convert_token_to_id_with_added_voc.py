def _convert_token_to_id_with_added_voc(self, token: str) ->int:
    index = self._tokenizer.token_to_id(token)
    if index is None:
        return self.unk_token_id
    return index
