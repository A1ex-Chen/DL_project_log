def export_model(model_id='', format='torchscript'):
    """Export a model to all formats."""
    assert format in export_fmts_hub(
        ), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(f'{HUB_API_ROOT}/v1/models/{model_id}/export', json={
        'format': format}, headers={'x-api-key': Auth().api_key})
    assert r.status_code == 200, f'{PREFIX}{format} export failure {r.status_code} {r.reason}'
    LOGGER.info(f'{PREFIX}{format} export started ✅')
