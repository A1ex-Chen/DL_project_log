@contextlib.contextmanager
def capture_logs_with_contextvars():
    cap = structlog.testing.LogCapture()
    old_processors = structlog.get_config()['processors']
    try:
        structlog.configure(processors=[structlog.contextvars.
            merge_contextvars, cap])
        yield cap.entries
    finally:
        structlog.configure(processors=old_processors)
