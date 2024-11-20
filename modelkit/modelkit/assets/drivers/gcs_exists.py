@retry(**GCS_RETRY_POLICY)
def exists(self, object_name):
    bucket = self.client.bucket(self.bucket)
    blob = bucket.blob(object_name)
    return blob.exists()
