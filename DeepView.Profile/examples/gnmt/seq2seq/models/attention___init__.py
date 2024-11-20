def __init__(self, query_size, key_size, num_units, normalize=False,
    batch_first=False, init_weight=0.1):
    """
        Constructor for the BahdanauAttention.

        :param query_size: feature dimension for query
        :param key_size: feature dimension for keys
        :param num_units: internal feature dimension
        :param normalize: whether to normalize energy term
        :param batch_first: if True batch size is the 1st dimension, if False
            the sequence is first and batch size is second
        :param init_weight: range for uniform initializer used to initialize
            Linear key and query transform layers and linear_att vector
        """
    super(BahdanauAttention, self).__init__()
    self.normalize = normalize
    self.batch_first = batch_first
    self.num_units = num_units
    self.linear_q = nn.Linear(query_size, num_units, bias=False)
    self.linear_k = nn.Linear(key_size, num_units, bias=False)
    nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
    nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)
    self.linear_att = Parameter(torch.Tensor(num_units))
    self.mask = None
    if self.normalize:
        self.normalize_scalar = Parameter(torch.Tensor(1))
        self.normalize_bias = Parameter(torch.Tensor(num_units))
    else:
        self.register_parameter('normalize_scalar', None)
        self.register_parameter('normalize_bias', None)
    self.reset_parameters(init_weight)
