@retry(**GCS_RETRY_POLICY)
def iterate_objects(self, prefix=None):
    bucket = self.client.bucket(self.bucket)
    for blob in bucket.list_blobs(prefix=prefix):
        yield blob.name
