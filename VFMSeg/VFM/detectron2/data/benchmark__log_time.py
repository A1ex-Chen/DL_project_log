def _log_time(self, msg, avg, all_times, distributed=False):
    percentiles = [np.percentile(all_times, k, interpolation='nearest') for
        k in [1, 5, 95, 99]]
    if not distributed:
        logger.info(
            f'{msg}: avg={1.0 / avg:.1f} it/s, p1={percentiles[0]:.2g}s, p5={percentiles[1]:.2g}s, p95={percentiles[2]:.2g}s, p99={percentiles[3]:.2g}s.'
            )
        return
    avg_per_gpu = comm.all_gather(avg)
    percentiles_per_gpu = comm.all_gather(percentiles)
    if comm.get_rank() > 0:
        return
    for idx, avg, percentiles in zip(count(), avg_per_gpu, percentiles_per_gpu
        ):
        logger.info(
            f'GPU{idx} {msg}: avg={1.0 / avg:.1f} it/s, p1={percentiles[0]:.2g}s, p5={percentiles[1]:.2g}s, p95={percentiles[2]:.2g}s, p99={percentiles[3]:.2g}s.'
            )
