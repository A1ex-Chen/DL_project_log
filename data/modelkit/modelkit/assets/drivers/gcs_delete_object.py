@retry(**GCS_RETRY_POLICY)
def delete_object(self, object_name):
    bucket = self.client.bucket(self.bucket)
    blob = bucket.blob(object_name)
    blob.delete()
