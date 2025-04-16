import torch
from ..multi_tensor_apply import multi_tensor_applier
from ._amp_state import _amp_state, master_params, maybe_print
from itertools import product



class LossScaler(object):
    warned_no_fused_kernel = False
    warned_unscaling_non_fp32_grad = False
    has_fused_kernel = False




    # unused_scale keeps some of the old API alive for hopefully a short time.

        # Defer to update_scale
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        # if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
        #     self._has_overflow = self._overflow_buf.item()



        # Defer to update_scale
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        # if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
        #     self._has_overflow = self._overflow_buf.item()


    # Separate so unscale() can be called more that once before updating.