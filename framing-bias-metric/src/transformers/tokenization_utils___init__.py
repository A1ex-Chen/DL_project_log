def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.added_tokens_encoder: Dict[str, int] = {}
    self.added_tokens_decoder: Dict[int, str] = {}
    self.unique_no_split_tokens: List[str] = []
