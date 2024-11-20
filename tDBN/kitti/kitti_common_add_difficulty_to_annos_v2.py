def add_difficulty_to_annos_v2(info):
    min_height = [40, 25, 25]
    max_occlusion = [0, 1, 2]
    max_trunc = [0.15, 0.3, 0.5]
    annos = info['annos']
    dims = annos['dimensions']
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = not (occlusion > max_occlusion[0] or height < min_height[0] or
        truncation > max_trunc[0])
    moderate_mask = not (occlusion > max_occlusion[1] or height <
        min_height[1] or truncation > max_trunc[1])
    hard_mask = not (occlusion > max_occlusion[2] or height < min_height[2] or
        truncation > max_trunc[2])
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)
    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff
