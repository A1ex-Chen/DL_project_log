@retry(**AZURE_RETRY_POLICY)
def download_object(self, object_name, destination_path):
    blob_client = self.client.get_blob_client(container=self.bucket, blob=
        object_name)
    if not blob_client.exists():
        logger.error('Object not found.', bucket=self.bucket, object_name=
            object_name)
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise errors.ObjectDoesNotExistError(driver=self, bucket=self.
            bucket, object_name=object_name)
    with open(destination_path, 'wb') as f:
        f.write(blob_client.download_blob().readall())
