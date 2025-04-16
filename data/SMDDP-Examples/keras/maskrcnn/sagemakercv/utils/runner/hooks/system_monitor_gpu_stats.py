def gpu_stats(self):
    return {'gpu_{}_{}'.format(i['index'], j): i[k] for i in gpustat.
        GPUStatCollection.new_query().jsonify()['gpus'] for j, k in zip([
        'temp', 'util', 'mem'], ['temperature.gpu', 'utilization.gpu',
        'memory.used'])}
