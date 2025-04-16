def __to_images_channel_last__(self, data: Union[torch.Tensor, List[torch.
    Tensor]], is_force: bool=False):
    if isinstance(data, list):
        return [self.__to_image_channel_last__(data=x, is_force=is_force) for
            x in data]
    elif isinstance(data, torch.Tensor):
        return self.__to_image_channel_last__(data=data, is_force=is_force)
    else:
        raise TypeError(
            f'Arguement data type, {type(data)},  is not supported, should be Union[torch.Tensor, List[torch.Tensor]]'
            )
