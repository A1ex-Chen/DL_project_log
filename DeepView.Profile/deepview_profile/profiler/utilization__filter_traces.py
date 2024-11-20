def _filter_traces(self, raw_slices):
    names_to_filter = ['profiler.py', 'built-in', 'torch/',
        'cudaDeviceSynchronize', 'typing.py', '<module>', 'os.py',
        '_collections', 'enum.py', 'numpy/', 'DataParallel', 'lib/', '.py']
    idx_to_filter = []
    for idx, item in enumerate(raw_slices['traceEvents']):
        if item['name'].startswith('torch/autograd') and item['name'].endswith(
            'backward'):
            continue
        for keyword in names_to_filter:
            if keyword in item['name']:
                idx_to_filter.append(idx)
        if item['name'].startswith('entry_point') and item['name'].endswith(
            'iteration'):
            idx_to_filter.append(idx)
        if item['name'].startswith('entry_point') and item['name'].endswith(
            'forward'):
            idx_to_filter.append(idx)
    filtered_slices = [event for idx, event in enumerate(raw_slices[
        'traceEvents']) if idx not in idx_to_filter]
    raw_slices['traceEvents'] = filtered_slices
    if logging.root.level == logging.DEBUG:
        with open('filtered_slices.json', 'wb') as f:
            f.write(orjson.dumps(raw_slices))
