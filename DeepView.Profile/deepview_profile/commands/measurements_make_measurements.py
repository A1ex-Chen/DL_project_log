def make_measurements(session, batch_size):
    session._batch_size = batch_size
    peak_usage_bytes = session.measure_peak_usage_bytes()
    thpt_msg = session.measure_throughput()
    return thpt_msg.samples_per_second, peak_usage_bytes
