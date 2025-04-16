def download_files(file_specs, num_threads=32, status_delay=0.2,
    timing_window=50, **download_kwargs):
    done_specs = {spec['file_path']: spec for spec in file_specs if os.path
        .isfile(spec['file_path'])}
    missing_specs = [spec for spec in file_specs if spec['file_path'] not in
        done_specs]
    files_total = len(file_specs)
    bytes_total = sum(spec['file_size'] for spec in file_specs)
    stats = dict(files_done=len(done_specs), bytes_done=sum(spec[
        'file_size'] for spec in done_specs.values()), lock=threading.Lock())
    if len(done_specs) == files_total:
        print('All files already downloaded -- skipping.')
        return
    spec_queue = queue.Queue()
    exception_queue = queue.Queue()
    for spec in missing_specs:
        spec_queue.put(spec)
    thread_kwargs = dict(spec_queue=spec_queue, exception_queue=
        exception_queue, stats=stats, download_kwargs=download_kwargs)
    for _thread_idx in range(min(num_threads, len(missing_specs))):
        threading.Thread(target=_download_thread, kwargs=thread_kwargs,
            daemon=True).start()
    bytes_unit, bytes_div = choose_bytes_unit(bytes_total)
    spinner = '/-\\|'
    timing = []
    while True:
        with stats['lock']:
            files_done = stats['files_done']
            bytes_done = stats['bytes_done']
        spinner = spinner[1:] + spinner[:1]
        timing = timing[max(len(timing) - timing_window + 1, 0):] + [(time.
            time(), bytes_done)]
        bandwidth = max((timing[-1][1] - timing[0][1]) / max(timing[-1][0] -
            timing[0][0], 1e-08), 0)
        bandwidth_unit, bandwidth_div = choose_bytes_unit(bandwidth)
        eta = format_time((bytes_total - bytes_done) / max(bandwidth, 1))
        print('\r%s %6.2f%% done  %d/%d files  %-13s  %-10s  ETA: %-7s ' %
            (spinner[0], bytes_done / bytes_total * 100, files_done,
            files_total, '%.2f/%.2f %s' % (bytes_done / bytes_div, 
            bytes_total / bytes_div, bytes_unit), '%.2f %s/s' % (bandwidth /
            bandwidth_div, bandwidth_unit), 'done' if bytes_total ==
            bytes_done else '...' if len(timing) < timing_window or 
            bandwidth == 0 else eta), end='', flush=True)
        if files_done == files_total:
            print()
            break
        try:
            exc_info = exception_queue.get(timeout=status_delay)
            raise exc_info[1].with_traceback(exc_info[2])
        except queue.Empty:
            pass
