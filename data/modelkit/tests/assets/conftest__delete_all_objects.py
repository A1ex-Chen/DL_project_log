def _delete_all_objects(mng):
    for object_name in mng.storage_provider.driver.iterate_objects(mng.
        storage_provider.prefix):
        mng.storage_provider.driver.delete_object(object_name)
