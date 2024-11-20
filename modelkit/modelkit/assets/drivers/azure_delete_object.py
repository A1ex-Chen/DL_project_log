@retry(**AZURE_RETRY_POLICY)
def delete_object(self, object_name):
    blob_client = self.client.get_blob_client(container=self.bucket, blob=
        object_name)
    blob_client.delete_blob()
