def tensor2imgs(self, images: [torch.Tensor, List[torch.Tensor]],
    cnvt_range: bool=True, is_detach: bool=False, to_device: Union[str,
    torch.device]=None, to_unint8: bool=False, to_channel: str=None, to_np:
    bool=False, to_pil: bool=False):
    trans = transforms.Compose(self.__reverse_pipe__(cnvt_range=cnvt_range,
        is_detach=is_detach, to_device=to_device, to_unint8=to_unint8,
        to_channel=to_channel, to_np=to_np, to_pil=to_pil))
    if isinstance(images, list):
        return [trans(img) for img in images]
    else:
        return trans(images)
