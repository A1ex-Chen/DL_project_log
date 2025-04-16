def __init__(self, url: str):
    """
        Keyword arguments:
        url: Fully qualified address of the Triton server - for e.g. grpc://localhost:8000
        """
    parsed_url = urlparse(url)
    if parsed_url.scheme == 'grpc':
        from tritonclient.grpc import InferenceServerClient, InferInput
        self.client = InferenceServerClient(parsed_url.netloc)
        model_repository = self.client.get_model_repository_index()
        self.model_name = model_repository.models[0].name
        self.metadata = self.client.get_model_metadata(self.model_name,
            as_json=True)

        def create_input_placeholders() ->typing.List[InferInput]:
            return [InferInput(i['name'], [int(s) for s in i['shape']], i[
                'datatype']) for i in self.metadata['inputs']]
    else:
        from tritonclient.http import InferenceServerClient, InferInput
        self.client = InferenceServerClient(parsed_url.netloc)
        model_repository = self.client.get_model_repository_index()
        self.model_name = model_repository[0]['name']
        self.metadata = self.client.get_model_metadata(self.model_name)

        def create_input_placeholders() ->typing.List[InferInput]:
            return [InferInput(i['name'], [int(s) for s in i['shape']], i[
                'datatype']) for i in self.metadata['inputs']]
    self._create_input_placeholders_fn = create_input_placeholders
