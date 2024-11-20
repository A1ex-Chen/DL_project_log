def _gather_and_numpify(self, tensors, name):
    """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
    if tensors is None:
        return
    if is_torch_tpu_available():
        tensors = nested_xla_mesh_reduce(tensors, name)
    elif self.args.local_rank != -1:
        tensors = distributed_concat(tensors)
    return nested_numpify(tensors)
