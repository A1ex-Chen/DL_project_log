@retry(**S3_RETRY_POLICY)
def upload_object(self, file_path, object_name):
    if self.aws_kms_key_id:
        self.client.upload_file(file_path, self.bucket, object_name,
            ExtraArgs={'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId':
            self.aws_kms_key_id})
    else:
        self.client.upload_file(file_path, self.bucket, object_name)
