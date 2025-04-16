def __to_image_channel_first__(self, data: torch.Tensor, is_force: bool=False):
    if self.__check_image_chennel__(data=data
        ) == MultiDataset.CHANNEL_LAST or is_force:
        return self.__force_image_channel_first__(data=data)
    return data
