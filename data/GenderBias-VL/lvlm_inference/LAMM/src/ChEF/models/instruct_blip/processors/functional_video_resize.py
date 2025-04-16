def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f'target size should be tuple (height, width), instead got {target_size}'
            )
    return torch.nn.functional.interpolate(clip, size=target_size, mode=
        interpolation_mode, align_corners=False)
