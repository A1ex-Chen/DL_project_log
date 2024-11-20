@typechecked
def __init__(self, weight_decay: Union[FloatTensorLike, Callable],
    decay_var_list: Optional[List]=None, *args, **kwargs):
    super().__init__(weight_decay, decay_var_list, *args, **kwargs)
