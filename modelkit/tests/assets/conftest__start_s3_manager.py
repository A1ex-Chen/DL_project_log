@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=
    stop_after_attempt(5), retry=retry_if_exception(lambda x: isinstance(x,
    Exception)), reraise=True)
def _start_s3_manager(working_dir):
    mng = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(prefix=f'test-assets-{uuid.uuid1().hex}', provider=
        's3', aws_default_region='us-east-1', bucket='test-assets',
        aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin',
        aws_session_token=None, s3_endpoint='http://127.0.0.1:9000'))
    mng.storage_provider.driver.client.create_bucket(Bucket='test-assets')
    return mng
