def start_tracking(self):
    super().start_tracking()
    self._hook_manager.attach_hooks_on_module(torch, lambda fn: 
        _is_callable_and_public(fn) and fn.__name__ not in
        BLACKLISTED_TORCH_METHODS, self._hook_creator)
    self._hook_manager.attach_hooks_on_module(torch.Tensor, lambda fn: 
        _is_callable_and_public(fn) and fn.__name__ != 'backward' and fn.
        __name__ not in BLACKLISTED_TENSOR_METHODS, self._hook_creator)
    self._hook_manager.attach_hooks_on_module(torch.Tensor,
        _is_callable_dunder, self._hook_creator)
    self._hook_manager.attach_hooks_on_module(torch.nn.functional,
        _is_callable_and_public, self._hook_creator)
    vf_module = (torch._VF if self._torch_version is None or self.
        _torch_version > OLD_VF_PATH_VERSION else torch.nn._VF)
    self._hook_manager.attach_hooks_on_module_using(vf_module, torch._C.
        _VariableFunctions, _is_callable_and_public, self._hook_creator)
