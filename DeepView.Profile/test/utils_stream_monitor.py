def stream_monitor(stream, callback=None):
    try:
        for line in stream:
            callback(line)
    except OSError:
        print(f'Closing listener for stream {stream}')
