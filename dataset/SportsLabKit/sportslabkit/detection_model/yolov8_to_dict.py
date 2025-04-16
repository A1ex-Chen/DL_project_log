def to_dict(res):
    if len(res) == 0:
        return [{}]
    return [{'bbox_left': r[0] - r[2] / 2, 'bbox_top': r[1] - r[3] / 2,
        'bbox_width': r[2], 'bbox_height': r[3], 'conf': r[4], 'class': r[5
        ]} for r in res]
