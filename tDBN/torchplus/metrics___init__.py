def __init__(self, dim=1, ignore_idx=-1, thresholds=0.5, use_sigmoid_score=
    False, encode_background_as_zeros=True):
    super().__init__()
    if not isinstance(thresholds, (list, tuple)):
        thresholds = [thresholds]
    self.register_buffer('prec_total', torch.FloatTensor(len(thresholds)).
        zero_())
    self.register_buffer('prec_count', torch.FloatTensor(len(thresholds)).
        zero_())
    self.register_buffer('rec_total', torch.FloatTensor(len(thresholds)).
        zero_())
    self.register_buffer('rec_count', torch.FloatTensor(len(thresholds)).
        zero_())
    self._ignore_idx = ignore_idx
    self._dim = dim
    self._thresholds = thresholds
    self._use_sigmoid_score = use_sigmoid_score
    self._encode_background_as_zeros = encode_background_as_zeros
