@s3_request
def s3_etag(url: str) ->Optional[str]:
    """Check ETag on S3 object."""
    s3_resource = boto3.resource('s3')
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag
