def __to_image_channel_last__(self, data: torch.Tensor, is_force: bool=False):
    if self.__check_image_chennel__(data=data
        ) == MultiDataset.CHANNEL_FIRST or is_force:
        return self.__force_images_channel_last__(data=data)
    return data
