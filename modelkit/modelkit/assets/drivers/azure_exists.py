@retry(**AZURE_RETRY_POLICY)
def exists(self, object_name):
    blob_client = self.client.get_blob_client(container=self.bucket, blob=
        object_name)
    return blob_client.exists()
