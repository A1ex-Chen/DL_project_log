def reset_model(model_id=''):
    """Reset a trained model to an untrained state."""
    r = requests.post(f'{HUB_API_ROOT}/model-reset', json={'modelId':
        model_id}, headers={'x-api-key': Auth().api_key})
    if r.status_code == 200:
        LOGGER.info(f'{PREFIX}Model reset successfully')
        return
    LOGGER.warning(f'{PREFIX}Model reset failure {r.status_code} {r.reason}')
