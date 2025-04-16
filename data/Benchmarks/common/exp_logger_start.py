def start(params, sys_desc):
    """Capture exp start"""
    exp_start = datetime.now()
    experiment_id = params['experiment_id']
    search_space = []
    for key, val in params.items():
        search_space.append('{}: {}'.format(key, val))
    msg = [{'experiment_id': experiment_id, 'start_time': str(exp_start),
        'system_description': sys_desc, 'search_space': search_space}]
    save('experiment_start.json', msg)
