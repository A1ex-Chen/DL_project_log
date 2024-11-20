def split_images_and_labels(data_root, save_dir, split='train', crop_sizes=
    (1024,), gaps=(200,)):
    """
    Split both images and labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - split
                - labels
                    - split
        and the output directory structure is:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / 'images' / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / 'labels' / split
    lb_dir.mkdir(parents=True, exist_ok=True)
    annos = load_yolo_dota(data_root, split=split)
    for anno in tqdm(annos, total=len(annos), desc=split):
        windows = get_windows(anno['ori_size'], crop_sizes, gaps)
        window_objs = get_window_obj(anno, windows)
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))
