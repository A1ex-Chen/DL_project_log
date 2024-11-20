def tokenize(self, text: str, pair: Optional[str]=None, add_special_tokens:
    bool=False, **kwargs) ->List[str]:
    return self.encode_plus(text=text, text_pair=pair, add_special_tokens=
        add_special_tokens, **kwargs).tokens()
