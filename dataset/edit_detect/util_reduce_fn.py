@staticmethod
def reduce_fn(reduce_type: str, dim: Union[int, Tuple[int], List[int]]=None,
    nd: int=None):
    if dim != None and nd != None:
        raise ValueError(
            'Arguement dim or nd should not be None inthe mean time')

    def get_dim(x, dim, nd):
        if nd != None:
            return list(range(len(x.shape)))[nd:]
        else:
            return dim
    if reduce_type == DistanceFn.REDUCE_MEAN:
        return lambda x: torch.mean(x, dim=get_dim(x=x, dim=dim, nd=nd))
    elif reduce_type == DistanceFn.REDUCE_SUM:
        return lambda x: torch.sum(x, dim=get_dim(x=x, dim=dim, nd=nd))
    else:
        raise NotImplementedError()
