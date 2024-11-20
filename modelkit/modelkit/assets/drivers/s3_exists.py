@retry(**S3_RETRY_POLICY)
def exists(self, object_name):
    try:
        self.client.head_object(Bucket=self.bucket, Key=object_name)
        return True
    except botocore.exceptions.ClientError:
        return False
