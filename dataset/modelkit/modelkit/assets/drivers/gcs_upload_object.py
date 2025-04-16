@retry(**GCS_RETRY_POLICY)
def upload_object(self, file_path, object_name):
    bucket = self.client.bucket(self.bucket)
    blob = bucket.blob(object_name)
    storage.blob._DEFAULT_CHUNKSIZE = 2097152
    storage.blob._MAX_MULTIPART_SIZE = 2097152
    with open(file_path, 'rb') as f:
        blob.upload_from_file(f)
