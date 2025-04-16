def __build_noises__(self, images: torch.Tensor) ->torch.Tensor:
    return torch.randn(size=images.shape, generator=self.__generator__
        ) * self.__epsilon_scale__
