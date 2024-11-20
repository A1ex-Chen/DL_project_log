def _print_panoptic_results(pq_res):
    headers = ['', 'PQ', 'SQ', 'RQ', '#categories']
    data = []
    for name in ['All', 'Things', 'Stuff']:
        row = [name] + [(pq_res[name][k] * 100) for k in ['pq', 'sq', 'rq']
            ] + [pq_res[name]['n']]
        data.append(row)
    table = tabulate(data, headers=headers, tablefmt='pipe', floatfmt='.3f',
        stralign='center', numalign='center')
    logger.info('Panoptic Evaluation Results:\n' + table)
