def start_tracking(self):
    super().start_tracking()
    self._hook_manager.attach_hook(torch.nn.Module, 'register_parameter',
        self._register_parameter_hook_creator)
