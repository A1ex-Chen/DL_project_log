def consumer():
    producer = threading.Thread(target=stream_producer)
    producer.start()
    while True:
        res = response_queue.get()
        if res is None:
            return
        yield res
