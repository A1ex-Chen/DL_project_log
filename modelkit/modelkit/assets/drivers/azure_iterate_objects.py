@retry(**AZURE_RETRY_POLICY)
def iterate_objects(self, prefix=None):
    container = self.client.get_container_client(self.bucket)
    for blob in container.list_blobs(name_starts_with=prefix):
        yield blob['name']
