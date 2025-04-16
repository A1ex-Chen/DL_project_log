def __check_images_chennel__(self, data: Union[torch.Tensor, List[torch.
    Tensor]], detected_channel_num: List[int]=[1, 3, 4]):
    if isinstance(data, list):
        return [self.__check_image_chennel__(data=x, detected_channel_num=
            detected_channel_num) for x in data]
    elif isinstance(data, torch.Tensor):
        return self.__check_image_chennel__(data=data, detected_channel_num
            =detected_channel_num)
    else:
        raise TypeError(
            f'Can only accept torch.Tensor or List[torch.Tensor] not {type(data)}'
            )
