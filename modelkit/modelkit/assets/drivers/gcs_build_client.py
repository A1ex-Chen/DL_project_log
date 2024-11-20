@staticmethod
def build_client(client_configuration: Dict[str, str]) ->Client:
    sa_path = client_configuration.get('service_account_path')
    if sa_path:
        return Client.from_service_account_json(sa_path)
    return Client()
