def measure_utilization(session):
    print('analysis: running measure_utilization()')
    yield session.measure_utilization()
    release_memory()
