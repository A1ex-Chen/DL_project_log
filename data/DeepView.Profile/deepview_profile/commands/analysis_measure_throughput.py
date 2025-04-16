def measure_throughput(session):
    print('analysis: running measure_throughput()')
    yield session.measure_throughput()
    release_memory()
