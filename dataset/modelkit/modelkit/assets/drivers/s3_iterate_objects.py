@retry(**S3_RETRY_POLICY)
def iterate_objects(self, prefix=None):
    paginator = self.client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix or '')
    for page in pages:
        for obj in page.get('Contents', []):
            yield obj['Key']
