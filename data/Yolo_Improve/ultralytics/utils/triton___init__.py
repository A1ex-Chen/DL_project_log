def __init__(self, url: str, endpoint: str='', scheme: str=''):
    """
        Initialize the TritonRemoteModel.

        Arguments may be provided individually or parsed from a collective 'url' argument of the form
            <scheme>://<netloc>/<endpoint>/<task_name>

        Args:
            url (str): The URL of the Triton server.
            endpoint (str): The name of the model on the Triton server.
            scheme (str): The communication scheme ('http' or 'grpc').
        """
    if not endpoint and not scheme:
        splits = urlsplit(url)
        endpoint = splits.path.strip('/').split('/')[0]
        scheme = splits.scheme
        url = splits.netloc
    self.endpoint = endpoint
    self.url = url
    if scheme == 'http':
        import tritonclient.http as client
        self.triton_client = client.InferenceServerClient(url=self.url,
            verbose=False, ssl=False)
        config = self.triton_client.get_model_config(endpoint)
    else:
        import tritonclient.grpc as client
        self.triton_client = client.InferenceServerClient(url=self.url,
            verbose=False, ssl=False)
        config = self.triton_client.get_model_config(endpoint, as_json=True)[
            'config']
    config['output'] = sorted(config['output'], key=lambda x: x.get('name'))
    type_map = {'TYPE_FP32': np.float32, 'TYPE_FP16': np.float16,
        'TYPE_UINT8': np.uint8}
    self.InferRequestedOutput = client.InferRequestedOutput
    self.InferInput = client.InferInput
    self.input_formats = [x['data_type'] for x in config['input']]
    self.np_input_formats = [type_map[x] for x in self.input_formats]
    self.input_names = [x['name'] for x in config['input']]
    self.output_names = [x['name'] for x in config['output']]
