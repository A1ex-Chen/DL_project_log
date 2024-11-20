def _stop_mocked_service_and_print_stderr(proc):
    proc.terminate()
    stderr = proc.stderr.read().decode()
    print(stderr, file=sys.stderr)
