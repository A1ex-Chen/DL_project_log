@retry(**GCS_RETRY_POLICY)
def download_object(self, object_name, destination_path):
    bucket = self.client.bucket(self.bucket)
    blob = bucket.blob(object_name)
    try:
        with open(destination_path, 'wb') as f:
            blob.download_to_file(f)
    except NotFound as e:
        logger.error('Object not found.', bucket=self.bucket, object_name=
            object_name)
        os.remove(destination_path)
        raise errors.ObjectDoesNotExistError(driver=self, bucket=self.
            bucket, object_name=object_name) from e
