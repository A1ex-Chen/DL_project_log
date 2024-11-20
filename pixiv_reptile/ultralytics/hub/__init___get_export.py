def get_export(model_id='', format='torchscript'):
    """Get an exported model dictionary with download URL."""
    assert format in export_fmts_hub(
        ), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(f'{HUB_API_ROOT}/get-export', json={'apiKey': Auth().
        api_key, 'modelId': model_id, 'format': format}, headers={
        'x-api-key': Auth().api_key})
    assert r.status_code == 200, f'{PREFIX}{format} get_export failure {r.status_code} {r.reason}'
    return r.json()
