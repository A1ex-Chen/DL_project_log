def chunk_read(response, chunk_size=8192, reporthook=None):
    total_size = response.info().get('Content-Length').strip()
    total_size = int(total_size)
    count = 0
    while 1:
        chunk = response.read(chunk_size)
        count += 1
        if not chunk:
            reporthook(count, total_size, total_size)
            break
        if reporthook:
            reporthook(count, chunk_size, total_size)
        yield chunk
