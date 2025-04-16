def reset_memory_hooks_state(self):
    """
        Reset the :obj:`mem_rss_diff` attribute of each module (see
        :func:`~transformers.modeling_utils.ModuleUtilsMixin.add_memory_hooks`).
        """
    for module in self.modules():
        module.mem_rss_diff = 0
        module.mem_rss_post_forward = 0
        module.mem_rss_pre_forward = 0
