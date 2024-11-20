def ddp_analysis(session):
    print('analysis: running ddp_computation()')
    yield session.ddp_computation()
    release_memory()
