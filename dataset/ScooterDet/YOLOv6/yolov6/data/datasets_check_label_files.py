@staticmethod
def check_label_files(args):
    img_path, lb_path = args
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ''
    try:
        if osp.exists(lb_path):
            nf = 1
            with open(lb_path, 'r') as f:
                labels = [x.split() for x in f.read().strip().splitlines() if
                    len(x)]
                labels = np.array(labels, dtype=np.float32)
            if len(labels):
                assert all(len(l) == 5 for l in labels
                    ), f'{lb_path}: wrong label format.'
                assert (labels >= 0).all(
                    ), f'{lb_path}: Label values error: all values in label file must > 0'
                assert (labels[:, 1:] <= 1).all(
                    ), f'{lb_path}: Label values error: all coordinates must be normalized'
                _, indices = np.unique(labels, axis=0, return_index=True)
                if len(indices) < len(labels):
                    labels = labels[indices]
                    msg += (
                        f'WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed'
                        )
                labels = labels.tolist()
            else:
                ne = 1
                labels = []
        else:
            nm = 1
            labels = []
        return img_path, labels, nc, nm, nf, ne, msg
    except Exception as e:
        nc = 1
        msg = f'WARNING: {lb_path}: ignoring invalid labels: {e}'
        return img_path, None, nc, nm, nf, ne, msg
