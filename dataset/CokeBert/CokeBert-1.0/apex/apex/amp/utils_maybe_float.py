def maybe_float(x, name='', verbose=False):
    if is_nested(x):
        return type(x)([maybe_float(y) for y in x])
    if not x.is_cuda or type_string(x) == 'FloatTensor':
        return x
    else:
        if verbose:
            print('Half->Float ({})'.format(name))
        return x.float()
