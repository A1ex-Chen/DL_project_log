@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=
    stop_after_attempt(5), retry=retry_if_exception(lambda x: isinstance(x,
    Exception)), reraise=True)
def _start_az_manager(working_dir):
    mng = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(prefix=f'test-assets-{uuid.uuid1().hex}', provider=
        'az', bucket='test-assets', connection_string=
        'DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;'
        ))
    mng.storage_provider.driver.client.create_container('test-assets')
    return mng
