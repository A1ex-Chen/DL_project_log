def __build_noise__(self):
    element_shape: List[int] = list(self.__loader__(data=self.__src__[0]).shape
        )
    if self.__noise_map_type__ == MultiDataset.NOISE_MAP_SAME:
        self.__noise__ = [x.squeeze(dim=0) for x in torch.randn(size=[self.
            __unique_src_n__] + element_shape, generator=self.__generator__
            ).split(1)]
        self.__noise__ = self.__repeat_data__(data=self.__noise__)
    elif self.__noise_map_type__ == MultiDataset.NOISE_MAP_DIFF:
        self.__noise__ = [x.squeeze(dim=0) for x in torch.randn(size=[self.
            __unique_src_n__ * self.__repeat__] + element_shape, generator=
            self.__generator__).split(1)]
    else:
        raise ValueError(
            f'Arguement noise_map_type is {self.__noise_map_type__}, should be {MultiDataset.NOISE_MAP_SAME} or {MultiDataset.NOISE_MAP_DIFF}'
            )
