@s3_request
def s3_get(url: str, temp_file: IO) ->None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource('s3')
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)
