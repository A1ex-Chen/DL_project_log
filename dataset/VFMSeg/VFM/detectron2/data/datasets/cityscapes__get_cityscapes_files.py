def _get_cityscapes_files(image_dir, gt_dir):
    files = []
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)
            suffix = 'leftImg8bit.png'
            assert basename.endswith(suffix), basename
            basename = basename[:-len(suffix)]
            instance_file = os.path.join(city_gt_dir, basename +
                'gtFine_instanceIds.png')
            label_file = os.path.join(city_gt_dir, basename +
                'gtFine_labelIds.png')
            json_file = os.path.join(city_gt_dir, basename +
                'gtFine_polygons.json')
            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), 'No images found in {}'.format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files
