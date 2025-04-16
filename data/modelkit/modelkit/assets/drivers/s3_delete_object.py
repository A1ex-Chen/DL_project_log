@retry(**S3_RETRY_POLICY)
def delete_object(self, object_name):
    self.client.delete_object(Bucket=self.bucket, Key=object_name)
