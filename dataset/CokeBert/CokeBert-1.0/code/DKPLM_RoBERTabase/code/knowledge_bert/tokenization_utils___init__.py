def __init__(self, max_len=None, **kwargs):
    self._bos_token = None
    self._eos_token = None
    self._unk_token = None
    self._sep_token = None
    self._pad_token = None
    self._cls_token = None
    self._mask_token = None
    self._pad_token_type_id = 0
    self._additional_special_tokens = []
    self.max_len = max_len if max_len is not None else int(1000000000000.0)
    self.padding_side = kwargs.pop('padding_side', self.padding_side)
    self.added_tokens_encoder = {}
    self.unique_added_tokens_encoder = set()
    self.added_tokens_decoder = {}
    self.init_inputs = ()
    self.init_kwargs = {}
    for key, value in kwargs.items():
        if key in self.SPECIAL_TOKENS_ATTRIBUTES:
            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all(isinstance(
                    t, str) or six.PY2 and isinstance(t, unicode) for t in
                    value)
            else:
                assert isinstance(value, str) or six.PY2 and isinstance(value,
                    unicode)
            setattr(self, key, value)
