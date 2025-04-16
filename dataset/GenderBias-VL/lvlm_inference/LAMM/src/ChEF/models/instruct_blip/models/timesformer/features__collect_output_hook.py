def _collect_output_hook(self, hook_id, *args):
    x = args[-1]
    if isinstance(x, tuple):
        x = x[0]
    self._feature_outputs[x.device][hook_id] = x
