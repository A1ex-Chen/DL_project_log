def _extract_repo_id_and_weights_name(pretrained_model_name_or_path):
    if not is_valid_url(pretrained_model_name_or_path):
        raise ValueError(
            'Invalid `pretrained_model_name_or_path` provided. Please set it to a valid URL.'
            )
    pattern = '([^/]+)/([^/]+)/(?:blob/main/)?(.+)'
    weights_name = None
    repo_id = None,
    for prefix in VALID_URL_PREFIXES:
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(
            prefix, '')
    match = re.match(pattern, pretrained_model_name_or_path)
    if not match:
        logger.warning(
            'Unable to identify the repo_id and weights_name from the provided URL.'
            )
        return repo_id, weights_name
    repo_id = f'{match.group(1)}/{match.group(2)}'
    weights_name = match.group(3)
    return repo_id, weights_name
