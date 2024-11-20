def main():
    import argparse
    parser = argparse.ArgumentParser(description='short_description')
    parser.add_argument('--dataset-dir', help=
        'Path to dataset directory where imagenet archives are stored and processed files will be saved.'
        , required=False, default=DATASETS_DIR)
    parser.add_argument('--target-size', help=
        'Size of target image. Format it as <width>,<height>.', required=
        False, default=','.join(map(str, TARGET_SIZE)))
    args = parser.parse_args()
    if args.dataset_dir is None:
        raise ValueError(
            'Please set $DATASETS_DIR env variable to point dataset dir with original dataset archives and where processed files should be stored. Alternatively provide --dataset-dir CLI argument'
            )
    datasets_dir = Path(args.dataset_dir)
    target_size = tuple(map(int, args.target_size.split(',')))
    image_archive_path = datasets_dir / IMAGE_ARCHIVE_FILENAME
    if not image_archive_path.exists():
        raise RuntimeError(
            f'There should be {IMAGE_ARCHIVE_FILENAME} file in {datasets_dir}.You need to download the dataset from http://www.image-net.org/download.'
            )
    devkit_archive_path = datasets_dir / DEVKIT_ARCHIVE_FILENAME
    if not devkit_archive_path.exists():
        raise RuntimeError(
            f'There should be {DEVKIT_ARCHIVE_FILENAME} file in {datasets_dir}.You need to download the dataset from http://www.image-net.org/download.'
            )
    with tarfile.open(devkit_archive_path, mode='r') as devkit_archive_file:
        labels_file = devkit_archive_file.extractfile(LABELS_REL_PATH)
        labels = list(map(int, labels_file.readlines()))
        meta_file = devkit_archive_file.extractfile(META_REL_PATH)
        idx_to_wnid = parse_meta_mat(meta_file)
        labels_wnid = [idx_to_wnid[idx] for idx in labels]
        available_wnids = sorted(set(labels_wnid))
        wnid_to_newidx = {wnid: new_cls for new_cls, wnid in enumerate(
            available_wnids)}
        labels = [wnid_to_newidx[wnid] for wnid in labels_wnid]
    output_dir = datasets_dir / IMAGENET_DIRNAME
    with tarfile.open(image_archive_path, mode='r') as image_archive_file:
        image_rel_paths = sorted(image_archive_file.getnames())
        for cls, image_rel_path in tqdm(zip(labels, image_rel_paths), total
            =len(image_rel_paths)):
            output_path = output_dir / str(cls) / image_rel_path
            original_image_file = image_archive_file.extractfile(image_rel_path
                )
            processed_image = _process_image(original_image_file, target_size)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed_image.save(output_path.as_posix())
