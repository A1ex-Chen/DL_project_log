def get_window_obj(anno, windows, iof_thr=0.7):
    """Get objects for each window."""
    h, w = anno['ori_size']
    label = anno['label']
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        iofs = bbox_iof(label[:, 1:], windows)
        return [label[iofs[:, i] >= iof_thr] for i in range(len(windows))]
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))
            ]
