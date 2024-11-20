def get_model(session, batch_size):
    session._batch_size = batch_size
    thpt_msg = session.measure_throughput()
    return (thpt_msg.peak_usage_bytes.slope, thpt_msg.peak_usage_bytes.bias), (
        thpt_msg.run_time_ms.slope, thpt_msg.run_time_ms.bias)
