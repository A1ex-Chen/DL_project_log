def checksum_file(logger, filename):
    """Read file, compute checksum, return it as a string."""
    import zlib
    start = time.time()
    chunk_size = 10 * 1024 * 1024
    total = 0
    with open(filename, 'rb') as fp:
        checksum = 0
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
    stop = time.time()
    MB = total / (1024 * 1024)
    duration = stop - start
    rate = MB / duration
    logger.info('checksummed: %0.3f MB in %.3f seconds (%.2f MB/s).' % (MB,
        duration, rate))
    return str(checksum)
