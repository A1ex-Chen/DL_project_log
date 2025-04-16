def _download_thread(spec_queue, exception_queue, stats, download_kwargs):
    with requests.Session() as session:
        while not spec_queue.empty():
            spec = spec_queue.get()
            try:
                download_file(session, spec, stats, **download_kwargs)
            except:
                exception_queue.put(sys.exc_info())
