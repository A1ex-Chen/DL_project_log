@retry(**S3_RETRY_POLICY)
def download_object(self, object_name, destination_path):
    try:
        with open(destination_path, 'wb') as f:
            self.client.download_fileobj(self.bucket, object_name, f)
    except botocore.exceptions.ClientError as e:
        logger.error('Object not found.', bucket=self.bucket, object_name=
            object_name)
        os.remove(destination_path)
        raise errors.ObjectDoesNotExistError(driver=self, bucket=self.
            bucket, object_name=object_name) from e
