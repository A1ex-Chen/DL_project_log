def is_sm_dist():
    """Check if environment variables are set for Sagemaker Data Distributed
    This has not been tested
    """
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if sm_training_env is None:
        return False
    sm_training_env = json.loads(sm_training_env)
    additional_framework_parameters = sm_training_env.get(
        'additional_framework_parameters', None)
    if not isinstance(additional_framework_parameters, dict):
        return False
    return bool(additional_framework_parameters.get(
        'sagemaker_distributed_dataparallel_enabled', False))
