def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
    super(FusedLayerNorm, self).__init__()
    global fused_layer_norm_cuda
    fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = normalized_shape,
    self.normalized_shape = torch.Size(normalized_shape)
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    if self.elementwise_affine:
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
    else:
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
    self.reset_parameters()
