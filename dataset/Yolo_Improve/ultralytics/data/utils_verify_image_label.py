def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = shape[1], shape[0]
        assert (shape[0] > 9) & (shape[1] > 9
            ), f'image size {shape} <10 pixels'
        assert im.format.lower(
            ) in IMG_FORMATS, f'invalid image format {im.format}. {FORMATS_HELP_MSG}'
        if im.format.lower() in {'jpg', 'jpeg'}:
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file,
                        'JPEG', subsampling=0, quality=100)
                    msg = (
                        f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'
                        )
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if
                    len(x)]
                if any(len(x) > 6 for x in lb) and not keypoint:
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-
                        1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1),
                        segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1
                        ] == 5 + nkpt * ndim, f'labels require {5 + nkpt * ndim} columns each'
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1
                        ] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    points = lb[:, 1:]
                assert points.max(
                    ) <= 1, f'non-normalized or out of bounds coordinates {points[points > 1]}'
                assert lb.min() >= 0, f'negative label values {lb[lb < 0]}'
                max_cls = lb[:, 0].max()
                assert max_cls <= num_cls, f'Label class {int(max_cls)} exceeds dataset class count {num_cls}. Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = (
                        f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
                        )
            else:
                ne = 1
                lb = np.zeros((0, 5 + nkpt * ndim if keypoint else 5),
                    dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5 + nkpt * ndim if keypoints else 5), dtype=
                np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[
                    ..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]],
                    axis=-1)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = (
            f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}')
        return [None, None, None, None, None, nm, nf, ne, nc, msg]
