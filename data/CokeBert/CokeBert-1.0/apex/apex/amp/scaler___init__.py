def __init__(self, loss_scale, init_scale=2.0 ** 16, scale_factor=2.0,
    scale_window=2000, min_loss_scale=None, max_loss_scale=2.0 ** 24):
    if loss_scale == 'dynamic':
        self.dynamic = True
        self._loss_scale = min(max_loss_scale, init_scale)
    else:
        self.dynamic = False
        self._loss_scale = loss_scale
    self._max_loss_scale = max_loss_scale
    self._min_loss_scale = min_loss_scale
    self._scale_seq_len = scale_window
    self._unskipped = 0
    self._has_overflow = False
    self._overflow_buf = torch.cuda.IntTensor([0])
    if multi_tensor_applier.available:
        import amp_C
        LossScaler.has_fused_kernel = multi_tensor_applier.available
        LossScaler.multi_tensor_scale_cuda = amp_C.multi_tensor_scale
        LossScaler.multi_tensor_axpby_cuda = amp_C.multi_tensor_axpby
    else:
        if not LossScaler.warned_no_fused_kernel:
            maybe_print(
                'Warning:  multi_tensor_applier fused unscale kernel is unavailable, possibly because apex was installed without --cuda_ext --cpp_ext. Using Python fallback.  Original ImportError was: '
                 + repr(multi_tensor_applier.import_err), True)
        LossScaler.has_fused_kernel = False
        LossScaler.warned_no_fused_kernel = True
