@staticmethod
def reshape_outputs(model, nc):
    """Update a TorchVision classification model to class count 'n' if required."""
    name, m = list((model.model if hasattr(model, 'model') else model).
        named_children())[-1]
    if isinstance(m, Classify):
        if m.linear.out_features != nc:
            m.linear = nn.Linear(m.linear.in_features, nc)
    elif isinstance(m, nn.Linear):
        if m.out_features != nc:
            setattr(model, name, nn.Linear(m.in_features, nc))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = len(types) - 1 - types[::-1].index(nn.Linear)
            if m[i].out_features != nc:
                m[i] = nn.Linear(m[i].in_features, nc)
        elif nn.Conv2d in types:
            i = len(types) - 1 - types[::-1].index(nn.Conv2d)
            if m[i].out_channels != nc:
                m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[
                    i].stride, bias=m[i].bias is not None)
