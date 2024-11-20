def habitat_predict(session):
    print('analysis: running deepview_predict()')
    yield session.habitat_predict()
    release_memory()
