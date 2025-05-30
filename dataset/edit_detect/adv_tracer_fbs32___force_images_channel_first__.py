def __force_images_channel_first__(self, data: Union[torch.Tensor, List[
    torch.Tensor]]):
    if isinstance(data, list):
        return [self.__force_image_channel_first__(data=x) for x in data]
    elif isinstance(data, torch.Tensor):
        return self.__force_image_channel_first__(data=data)
    else:
        raise TypeError(
            f'Arguement data type, {type(data)},  is not supported, should be Union[torch.Tensor, List[torch.Tensor]]'
            )
