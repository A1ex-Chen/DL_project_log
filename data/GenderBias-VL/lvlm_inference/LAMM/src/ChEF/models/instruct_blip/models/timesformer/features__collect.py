def _collect(self, x) ->Dict[str, torch.Tensor]:
    out = OrderedDict()
    for name, module in self.items():
        x = module(x)
        if name in self.return_layers:
            out_id = self.return_layers[name]
            if isinstance(x, (tuple, list)):
                out[out_id] = torch.cat(x, 1) if self.concat else x[0]
            else:
                out[out_id] = x
    return out
