def get_performance_limits(memory_info, throughput_info):
    max_capacity_batch_size = memory_info.usage_model_mb.inverse(memory_info
        .max_capacity_mb)
    max_capacity_throughput = (max_capacity_batch_size / throughput_info.
        runtime_model_ms.evaluate(max_capacity_batch_size) * 1000)
    max_throughput_batch_size = throughput_info.batch_from_throughput(
        throughput_info.max_throughput)
    thpt_limits = max_throughput_batch_size, throughput_info.max_throughput
    mem_limits = max_capacity_batch_size, max_capacity_throughput
    limits = min(thpt_limits, mem_limits, key=lambda tup: tup[0])
    return PerformanceLimits(max_batch_size=limits[0], throughput_limit=
        limits[1])
