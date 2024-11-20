def stop_tracking(self):
    super().stop_tracking()
    self._hook_manager.remove_hooks()
