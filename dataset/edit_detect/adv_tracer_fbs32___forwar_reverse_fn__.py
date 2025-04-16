def __forwar_reverse_fn__(self):
    norm_fn = [transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1,
        vmin_out=self.__vmin_out__, vmax_out=self.__vmax_out__, x=x)),
        transforms.Lambda(lambda x: normalize(vmin_in=self.__vmin_out__,
        vmax_in=self.__vmax_out__, vmin_out=0, vmax_out=1, x=x))]
    arithm_fn = [transforms.Lambda(lambda x: (x - 0.5) * 2), transforms.
        Lambda(lambda x: x / 2 + 0.5)]
    torch_norm_fn = [transforms.Normalize(mean=0, std=0.5), transforms.
        Lambda(lambda x: x / 2 + 0.5)]
    return arithm_fn
