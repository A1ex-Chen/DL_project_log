def plot_query_result(similar_set, plot_labels=True):
    """
    Plot images from the similar set.

    Args:
        similar_set (list): Pyarrow or pandas object containing the similar data points
        plot_labels (bool): Whether to plot labels or not
    """
    import pandas
    similar_set = similar_set.to_dict(orient='list') if isinstance(similar_set,
        pandas.DataFrame) else similar_set.to_pydict()
    empty_masks = [[[]]]
    empty_boxes = [[]]
    images = similar_set.get('im_file', [])
    bboxes = similar_set.get('bboxes', []) if similar_set.get('bboxes'
        ) is not empty_boxes else []
    masks = similar_set.get('masks') if similar_set.get('masks')[0
        ] != empty_masks else []
    kpts = similar_set.get('keypoints') if similar_set.get('keypoints')[0
        ] != empty_masks else []
    cls = similar_set.get('cls', [])
    plot_size = 640
    imgs, batch_idx, plot_boxes, plot_masks, plot_kpts = [], [], [], [], []
    for i, imf in enumerate(images):
        im = cv2.imread(imf)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        r = min(plot_size / h, plot_size / w)
        imgs.append(LetterBox(plot_size, center=False)(image=im).transpose(
            2, 0, 1))
        if plot_labels:
            if len(bboxes) > i and len(bboxes[i]) > 0:
                box = np.array(bboxes[i], dtype=np.float32)
                box[:, [0, 2]] *= r
                box[:, [1, 3]] *= r
                plot_boxes.append(box)
            if len(masks) > i and len(masks[i]) > 0:
                mask = np.array(masks[i], dtype=np.uint8)[0]
                plot_masks.append(LetterBox(plot_size, center=False)(image=
                    mask))
            if len(kpts) > i and kpts[i] is not None:
                kpt = np.array(kpts[i], dtype=np.float32)
                kpt[:, :, :2] *= r
                plot_kpts.append(kpt)
        batch_idx.append(np.ones(len(np.array(bboxes[i], dtype=np.float32))
            ) * i)
    imgs = np.stack(imgs, axis=0)
    masks = np.stack(plot_masks, axis=0) if plot_masks else np.zeros(0,
        dtype=np.uint8)
    kpts = np.concatenate(plot_kpts, axis=0) if plot_kpts else np.zeros((0,
        51), dtype=np.float32)
    boxes = xyxy2xywh(np.concatenate(plot_boxes, axis=0)
        ) if plot_boxes else np.zeros(0, dtype=np.float32)
    batch_idx = np.concatenate(batch_idx, axis=0)
    cls = np.concatenate([np.array(c, dtype=np.int32) for c in cls], axis=0)
    return plot_images(imgs, batch_idx, cls, bboxes=boxes, masks=masks,
        kpts=kpts, max_subplots=len(images), save=False, threaded=False)
