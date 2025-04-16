def http_user_agent(user_agent: Union[Dict, str, None]=None) ->str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = 'transformers/{}; python/{}'.format(__version__, sys.version.split
        ()[0])
    if is_torch_available():
        ua += '; torch/{}'.format(torch.__version__)
    if is_tf_available():
        ua += '; tensorflow/{}'.format(tf.__version__)
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join('{}/{}'.format(k, v) for k, v in user_agent.
            items())
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    return ua
