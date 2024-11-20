@property
def client(self):
    return self._client or self.build_client(self.client_configuration)
