def load_yolo_dota(data_root, split='train'):
    """
    Load DOTA dataset.

    Args:
        data_root (str): Data root.
        split (str): The split data set, could be train or val.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {'train', 'val'
        }, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / 'images' / split
    assert im_dir.exists(
        ), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / 'images' / split / '*'))
    lb_files = img2label_paths(im_files)
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos
