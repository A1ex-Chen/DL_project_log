def _download_object_or_prefix(driver, object_name, destination_dir):
    asset_path = os.path.join(destination_dir, 'myasset')
    try:
        driver.download_object(object_name=object_name, destination_path=
            asset_path)
    except ObjectDoesNotExistError:
        paths = [path for path in driver.iterate_objects(prefix=object_name)]
        if not paths:
            raise
        os.mkdir(asset_path)
        for path in paths:
            sub_object_name = path.split('/')[-1]
            driver.download_object(object_name=object_name + '/' +
                sub_object_name, destination_path=os.path.join(asset_path,
                sub_object_name))
    return asset_path
