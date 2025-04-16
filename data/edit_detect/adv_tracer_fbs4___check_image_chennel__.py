def __check_image_chennel__(self, data: torch.Tensor, detected_channel_num:
    List[int]=[1, 3, 4]):
    if data.shape[-1] in detected_channel_num:
        return MultiDataset.CHANNEL_LAST
    elif data.shape[-3] in detected_channel_num:
        return MultiDataset.CHANNEL_FIRST
    else:
        raise ValueError(f'Can only detect channel first or last')
