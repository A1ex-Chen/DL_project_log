@retry(**AZURE_RETRY_POLICY)
def upload_object(self, file_path, object_name):
    blob_client = self.client.get_blob_client(container=self.bucket, blob=
        object_name)
    if blob_client.exists():
        self.delete_object(object_name)
    with open(file_path, 'rb') as f:
        blob_client.upload_blob(f)
