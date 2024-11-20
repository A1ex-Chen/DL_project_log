def prepare_mask_coef_by_statistics(num_frames: int, cond_frame: int,
    motion_scale: int):
    assert num_frames > 0, 'video_length should be greater than 0'
    assert num_frames > cond_frame, 'video_length should be greater than cond_frame'
    range_list = RANGE_LIST
    assert motion_scale < len(range_list
        ), f'motion_scale type{motion_scale} not implemented'
    coef = range_list[motion_scale]
    coef = coef + [coef[-1]] * (num_frames - len(coef))
    order = [abs(i - cond_frame) for i in range(num_frames)]
    coef = [coef[order[i]] for i in range(num_frames)]
    return coef
