def end(experiment_id):
    """Capture exp end"""
    exp_end = datetime.now()
    msg = [{'experiment_id': experiment_id, 'status': {'set': 'Finished'},
        'end_time': {'set': str(exp_end)}}]
    save('experiment_end.json', msg)
