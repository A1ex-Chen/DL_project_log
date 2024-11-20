def log_scalars(self, scalar_dict, step, prefix=None):
    if self.writer is None:
        return
    for k in scalar_dict:
        v = scalar_dict[k]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().item()
        if prefix is not None:
            k = prefix + k
        self.writer.add_scalar(k, v, step)
