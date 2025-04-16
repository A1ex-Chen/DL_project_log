def __image_channel__(self, data: Union[torch.Tensor, List[torch.Tensor]]):
    return data.permute(1, 2, 0)
