@staticmethod
def _parse_identifier(identifier):
    """
        Parses the given identifier to determine the type of identifier and extract relevant components.

        The method supports different identifier formats:
            - A HUB URL, which starts with HUB_WEB_ROOT followed by '/models/'
            - An identifier containing an API key and a model ID separated by an underscore
            - An identifier that is solely a model ID of a fixed length
            - A local filename that ends with '.pt' or '.yaml'

        Args:
            identifier (str): The identifier string to be parsed.

        Returns:
            (tuple): A tuple containing the API key, model ID, and filename as applicable.

        Raises:
            HUBModelError: If the identifier format is not recognized.
        """
    api_key, model_id, filename = None, None, None
    if identifier.startswith(f'{HUB_WEB_ROOT}/models/'):
        model_id = identifier.split(f'{HUB_WEB_ROOT}/models/')[-1]
    else:
        parts = identifier.split('_')
        if len(parts) == 2 and len(parts[0]) == 42 and len(parts[1]) == 20:
            api_key, model_id = parts
        elif len(parts) == 1 and len(parts[0]) == 20:
            model_id = parts[0]
        elif identifier.endswith('.pt') or identifier.endswith('.yaml'):
            filename = identifier
        else:
            raise HUBModelError(
                f"model='{identifier}' could not be parsed. Check format is correct. Supported formats are Ultralytics HUB URL, apiKey_modelId, modelId, local pt or yaml file."
                )
    return api_key, model_id, filename
