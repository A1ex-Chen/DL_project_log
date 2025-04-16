def _convert_id_to_token(self, index: int) ->Optional[str]:
    return self._tokenizer.id_to_token(int(index))
