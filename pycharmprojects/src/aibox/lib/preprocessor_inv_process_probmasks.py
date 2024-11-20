@classmethod
def inv_process_probmasks(cls, process_dict: Dict[str, Any], probmasks: Tensor
    ) ->Tensor:
    assert probmasks.ndim == 4
    if probmasks.shape[0] > 0:
        right_pad, bottom_pad = process_dict[cls.PROCESS_KEY_RIGHT_PAD
            ], process_dict[cls.PROCESS_KEY_BOTTOM_PAD]
        inv_probmasks = F.pad(input=probmasks, pad=[0, -right_pad, 0, -
            bottom_pad])
        origin_size = process_dict[cls.PROCESS_KEY_ORIGIN_HEIGHT
            ], process_dict[cls.PROCESS_KEY_ORIGIN_WIDTH]
        inv_probmasks = F.interpolate(input=inv_probmasks, size=origin_size,
            mode='bilinear', align_corners=True)
    else:
        inv_probmasks = probmasks
    return inv_probmasks
