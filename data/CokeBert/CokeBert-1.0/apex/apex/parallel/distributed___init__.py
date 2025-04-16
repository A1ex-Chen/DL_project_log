def __init__(self, module, message_size=10000000, delay_allreduce=False,
    shared_param=None, allreduce_trigger_params=None,
    retain_allreduce_buffers=False, allreduce_always_fp32=False,
    gradient_average=True, gradient_predivide_factor=1.0):
    super(DistributedDataParallel, self).__init__()
    if hasattr(dist, 'get_backend'):
        self._backend = dist.get_backend()
        if hasattr(dist, 'DistBackend'):
            self.backend_enum_holder = dist.DistBackend
        else:
            self.backend_enum_holder = dist.Backend
    else:
        self._backend = dist._backend
        self.backend_enum_holder = dist.dist_backend
    self.warn_on_half = (True if self._backend == self.backend_enum_holder.
        GLOO else False)
    if shared_param is not None:
        raise ValueError(
            'shared_param is no longer supported as an option.  It was misleadingly named from the start.  It turns out overlapping communication with computation should work fine with shared parameters.  If you still wish to delay communication to the end of the backward pass, use delay_allreduce=True|False instead.'
            )
    self.world_size = float(dist.get_world_size())
    self.retain_allreduce_buffers = retain_allreduce_buffers
    self.allreduce_always_fp32 = allreduce_always_fp32
    self.gradient_average = gradient_average
    self.gradient_predivide_factor = gradient_predivide_factor
    self.custom_allreduce_triggers = False
    if allreduce_trigger_params is not None:
        if delay_allreduce:
            raise ValueError(
                'Setting allreduce_trigger_params is only valid if delay_allreduce=False.'
                )
        self.custom_allreduce_triggers = True
        self.allreduce_trigger_params = set([id(param) for param in
            allreduce_trigger_params])
    self.delay_allreduce = delay_allreduce
    self.message_size = message_size
    self.reduction_stream = torch.cuda.Stream()
    self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False
        )
    self.module = module
    self._disable_allreduce = False
    if self._backend == self.backend_enum_holder.NCCL:
        for param in self.module.parameters():
            assert param.is_cuda, 'NCCL backend only supports model parameters to be on GPU.'
    self.active_params = []
    self.param_type_to_tmp_i = {'torch.cuda.HalfTensor': 0,
        'torch.cuda.FloatTensor': 1, 'torch.cuda.DoubleTensor': 2}
    if multi_tensor_applier.available:
        import amp_C
        self.multi_tensor_scale = amp_C.multi_tensor_scale
        self._overflow_buf = torch.cuda.IntTensor([0])
    self.create_hooks()
    flat_dist_call([param.data for param in self.module.parameters()], dist
        .broadcast, (0,))
