def set_runtimes_from_cache(self, cached_info_map):
    """
        Used to set the runtimes from cache for when the parsed code has not
        changed.
        """
    for bound_name, op_info in self.operations.items():
        cached_op_info = cached_info_map.get_operation_info_by_bound_name(
            bound_name)
        op_info.runtime_us = cached_op_info.runtime_us
