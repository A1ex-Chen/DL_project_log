def energy_compute(session):
    print('analysis: running energy_compute()')
    yield session.energy_compute()
    release_memory()
