def http_user_agent(user_agent: Union[Dict, str, None]=None) ->str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f'diffusers; python/{sys.version.split()[0]}; session_id/{SESSION_ID}'
    if DISABLE_TELEMETRY or HF_HUB_OFFLINE:
        return ua + '; telemetry/off'
    if is_torch_available():
        ua += f'; torch/{_torch_version}'
    if is_flax_available():
        ua += f'; jax/{_jax_version}'
        ua += f'; flax/{_flax_version}'
    if is_onnx_available():
        ua += f'; onnxruntime/{_onnxruntime_version}'
    if os.environ.get('DIFFUSERS_IS_CI', '').upper() in ENV_VARS_TRUE_VALUES:
        ua += '; is_ci/true'
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join(f'{k}/{v}' for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    return ua
