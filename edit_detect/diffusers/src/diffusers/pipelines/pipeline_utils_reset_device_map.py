def reset_device_map(self):
    """
        Resets the device maps (if any) to None.
        """
    if self.hf_device_map is None:
        return
    else:
        self.remove_all_hooks()
        for name, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                component.to('cpu')
        self.hf_device_map = None
