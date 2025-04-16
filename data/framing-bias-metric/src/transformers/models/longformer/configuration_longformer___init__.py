def __init__(self, attention_window: Union[List[int], int]=512,
    sep_token_id: int=2, **kwargs):
    super().__init__(sep_token_id=sep_token_id, **kwargs)
    self.attention_window = attention_window
