def add_memory_hooks(self):
    """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a :obj:`mem_rss_diff` attribute for each module and can be reset to
        zero with :obj:`model.reset_memory_hooks_state()`.
        """
    for module in self.modules():
        module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
        module.register_forward_hook(self._hook_rss_memory_post_forward)
    self.reset_memory_hooks_state()
