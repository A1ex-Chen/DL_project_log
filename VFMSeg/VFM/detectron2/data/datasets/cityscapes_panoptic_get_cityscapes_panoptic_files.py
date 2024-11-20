def get_cityscapes_panoptic_files(image_dir, gt_dir, json_info):
    files = []
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    image_dict = {}
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)
            suffix = '_leftImg8bit.png'
            assert basename.endswith(suffix), basename
            basename = os.path.basename(basename)[:-len(suffix)]
            image_dict[basename] = image_file
    for ann in json_info['annotations']:
        image_file = image_dict.get(ann['image_id'], None)
        assert image_file is not None, 'No image {} found for annotation {}'.format(
            ann['image_id'], ann['file_name'])
        label_file = os.path.join(gt_dir, ann['file_name'])
        segments_info = ann['segments_info']
        files.append((image_file, label_file, segments_info))
    assert len(files), 'No images found in {}'.format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files
