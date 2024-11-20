def __init__(self, model, num_classes=80, num_layers=3, anchors=1, device=
    torch.device('cuda:0'), return_int=False, scale_exact=False,
    force_no_pad=False, is_end2end=False):
    self.return_int = return_int
    self.scale_exact = scale_exact
    self.force_no_pad = force_no_pad
    self.is_end2end = is_end2end
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    self.logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(self.logger, namespace='')
    self.runtime = trt.Runtime(self.logger)
    with open(model, 'rb') as f:
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
    self.input_shape = get_input_shape(self.engine)
    self.bindings = OrderedDict()
    self.input_names = list()
    self.output_names = list()
    for index in range(self.engine.num_bindings):
        name = self.engine.get_binding_name(index)
        if self.engine.binding_is_input(index):
            self.input_names.append(name)
        else:
            self.output_names.append(name)
        dtype = trt.nptype(self.engine.get_binding_dtype(index))
        shape = tuple(self.engine.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(
            device)
        self.bindings[name] = Binding(name, dtype, shape, data, int(data.
            data_ptr()))
    self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.
        items())
    self.context = self.engine.create_execution_context()
    assert self.engine
    assert self.context
    self.nc = num_classes
    self.no = num_classes + 5
    self.nl = num_layers
    if isinstance(anchors, (list, tuple)):
        self.na = len(anchors[0]) // 2
    else:
        self.na = anchors
    self.anchors = anchors
    self.grid = [torch.zeros(1, device=device)] * num_layers
    self.prior_prob = 0.01
    self.inplace = True
    stride = [8, 16, 32]
    self.stride = torch.tensor(stride, device=device)
    self.shape = [80, 40, 20]
    self.device = device
