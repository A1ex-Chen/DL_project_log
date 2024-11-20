def _decode(self, token_ids: Union[int, List[int]], skip_special_tokens:
    bool=False, clean_up_tokenization_spaces: bool=True, **kwargs) ->str:
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    text = self._tokenizer.decode(token_ids, skip_special_tokens=
        skip_special_tokens)
    if clean_up_tokenization_spaces:
        clean_text = self.clean_up_tokenization(text)
        return clean_text
    else:
        return text
