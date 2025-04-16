def __reverse_pipe__(self, cnvt_range: bool=True, is_detach: bool=False,
    to_device: Union[str, torch.device]=None, to_unint8: bool=False,
    to_channel: str=None, to_np: bool=False, to_pil: bool=False):
    trans = [self.__forwar_reverse_fn__()[1]]
    detach_trans = [transforms.Lambda(lambda x: x.detach())]
    cpu_trans = [transforms.Lambda(lambda x: x.cpu())]
    unint8_trans = [transforms.Lambda(lambda x: x.mul_(255).add_(0.5).
        clamp_(0, 255).to(torch.uint8))]
    channel_last_trans = [transforms.Lambda(lambda x: self.
        __to_images_channel_last__(data=x))]
    channel_first_trans = [transforms.Lambda(lambda x: self.
        __to_images_channel_first__(data=x))]
    np_trans = [transforms.Lambda(lambda x: x.numpy())]
    pil_trans = [transforms.ToPILImage(mode='RGB')]
    if to_pil:
        return (trans + detach_trans + cpu_trans + unint8_trans +
            channel_last_trans + np_trans + pil_trans)
    res_trans = []
    if cnvt_range:
        res_trans = res_trans + trans
    if detach_trans:
        res_trans = res_trans + detach_trans
    if to_device is not None:
        res_trans = res_trans + [transforms.Lambda(lambda x: x.to(to_device))]
    if to_unint8:
        res_trans = res_trans + unint8_trans
    if to_channel == MultiDataset.CHANNEL_FIRST:
        res_trans = res_trans + channel_first_trans
    elif to_channel == MultiDataset.CHANNEL_LAST:
        res_trans = res_trans + channel_last_trans
    if to_np:
        res_trans = res_trans + detach_trans + cpu_trans + np_trans
    return res_trans
