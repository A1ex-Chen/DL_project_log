def __init__(self, do_lowercase=False, **kwargs):
    super().__init__(**kwargs)
    self.do_lowercase = do_lowercase
    self.do_lowercase_and_remove_accent = False
