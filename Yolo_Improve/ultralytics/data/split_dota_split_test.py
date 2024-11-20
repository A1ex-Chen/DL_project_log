def split_test(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    Split create_self_data set of DOTA, labels are not included within this set.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - create_self_data
        and the output directory structure is:
            - save_dir
                - images
                    - create_self_data
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    save_dir = Path(save_dir) / 'images' / 'create_self_data'
    save_dir.mkdir(parents=True, exist_ok=True)
    im_dir = Path(data_root) / 'images' / 'create_self_data'
    assert im_dir.exists(
        ), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(im_dir / '*'))
    for im_file in tqdm(im_files, total=len(im_files), desc='create_self_data'
        ):
        w, h = exif_size(Image.open(im_file))
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        im = cv2.imread(im_file)
        name = Path(im_file).stem
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f'{name}__{x_stop - x_start}__{x_start}___{y_start}'
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f'{new_name}.jpg'), patch_im)
