def normalize(self, x: str) ->str:
    """Cover moses empty string edge case. They return empty list for '' input!"""
    return self.punc_normalizer(x) if x else ''
