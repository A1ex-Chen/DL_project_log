def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """
    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. You need to clear the cache file if your dataset has been modified."
                )
        else:
            logger.info(
                f"Converting annotations of dataset '{dataset_name}' to COCO format ...)"
                )
            coco_dict = convert_to_coco_dict(dataset_name)
            logger.info(
                f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + '.tmp'
            with PathManager.open(tmp_file, 'w') as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)
