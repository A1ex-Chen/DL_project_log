def model_list(self) ->List[ModelInfo]:
    """
        Get the public list of all the models on huggingface, including the community models
        """
    path = '{}/api/models'.format(self.endpoint)
    r = requests.get(path)
    r.raise_for_status()
    d = r.json()
    return [ModelInfo(**x) for x in d]
