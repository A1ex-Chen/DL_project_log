def crop_and_save(anno, windows, window_objs, im_dir, lb_dir):
    """
    Crop images and save new labels.

    Args:
        anno (dict): Annotation dict, including `filepath`, `label`, `ori_size` as its keys.
        windows (list): A list of windows coordinates.
        window_objs (list): A list of labels inside each window.
        im_dir (str): The output directory path of images.
        lb_dir (str): The output directory path of labels.

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
    im = cv2.imread(anno['filepath'])
    name = Path(anno['filepath']).stem
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f'{name}__{x_stop - x_start}__{x_start}___{y_start}'
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]
        cv2.imwrite(str(Path(im_dir) / f'{new_name}.jpg'), patch_im)
        label = window_objs[i]
        if len(label) == 0:
            continue
        label[:, 1::2] -= x_start
        label[:, 2::2] -= y_start
        label[:, 1::2] /= pw
        label[:, 2::2] /= ph
        with open(Path(lb_dir) / f'{new_name}.txt', 'w') as f:
            for lb in label:
                formatted_coords = ['{:.6g}'.format(coord) for coord in lb[1:]]
                f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")
