def download_object(self, object_name, destination_path):
    object_path = os.path.join(self.bucket, *object_name.split('/'))
    if not os.path.isfile(object_path):
        logger.error('Object not found.', bucket=self.bucket, object_name=
            object_name)
        raise errors.ObjectDoesNotExistError(driver=self, bucket=self.
            bucket, object_name=object_name)
    with open(object_path, 'rb') as fsrc:
        with open(destination_path, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
