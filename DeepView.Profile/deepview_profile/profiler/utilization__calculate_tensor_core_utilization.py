def _calculate_tensor_core_utilization(self, filepath):
    kernelDict = {'tensorTime': 0, 'noTensorTime': 0, 'totalTime': 0}
    with open(filepath, 'r') as f:
        data = orjson.loads(f.read())
    for event in data['traceEvents']:
        if event.get('cat') and event['cat'] == 'kernel':
            if event['name'] in TC_Allowlist:
                kernelDict['tensorTime'] += event['dur']
            else:
                kernelDict['noTensorTime'] += event['dur']
    totalTime = kernelDict['tensorTime'] + kernelDict['noTensorTime']
    if logging.root.level == logging.DEBUG:
        logger.debug(
            f"""Tensor time: {kernelDict['tensorTime']} perc {round(kernelDict['tensorTime'] / totalTime * 100, 2)}
"""
            )
        logger.debug(
            f"""No Tensor time: {kernelDict['noTensorTime']} perc {round(kernelDict['noTensorTime'] / totalTime * 100, 2)}
"""
            )
        logger.debug(f'totalTime: {totalTime}\n')
    return round(kernelDict['tensorTime'] / totalTime * 100, 2)
