def gather(self, outputs: List[Tuple], target_device: int, dim: int=0):

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Bunch):
            return Bunch([o.to(torch.device('cuda', target_device)) for out in
                outputs for o in out])
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in
                out)
        return type(out)(map(gather_map, zip(*outputs)))
    try:
        gathred_outputs = gather_map(outputs)
    finally:
        gather_map = None
    return gathred_outputs
