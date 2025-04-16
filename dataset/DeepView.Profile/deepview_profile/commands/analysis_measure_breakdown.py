def measure_breakdown(session, nvml):
    print('analysis: running measure_breakdown()')
    yield session.measure_breakdown(nvml)
    release_memory()
