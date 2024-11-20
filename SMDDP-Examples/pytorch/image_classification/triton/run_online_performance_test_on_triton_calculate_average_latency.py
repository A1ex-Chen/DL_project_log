def calculate_average_latency(r):
    avg_sum_fields = ['Client Send', 'Network+Server Send/Recv',
        'Server Queue', 'Server Compute', 'Server Compute Input',
        'Server Compute Infer', 'Server Compute Output', 'Client Recv']
    avg_latency = sum([int(r.get(f, 0)) for f in avg_sum_fields])
    return avg_latency
