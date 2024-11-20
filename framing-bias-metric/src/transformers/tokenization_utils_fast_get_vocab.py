def get_vocab(self) ->Dict[str, int]:
    return self._tokenizer.get_vocab(with_added_tokens=True)
