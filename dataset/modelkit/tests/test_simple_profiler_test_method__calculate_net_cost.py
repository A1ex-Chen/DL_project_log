def test_method__calculate_net_cost():
    model_library = ModelLibrary(models=[ModelA, ModelB, ModelC, ModelD,
        Pipeline])
    pipeline = model_library.get('pipeline')
    item = {'abc': 123}
    profiler = SimpleProfiler(pipeline)
    _ = pipeline.predict(item)
    profiler.net_durations = {'model_a': [1.0, 1.01], 'model_b': [0.5],
        'model_c': [0.7], 'model_d': [0.1], 'pipeline': [0.2]}
    assert math.isclose(profiler._calculate_net_cost(1.51, {'model_a': 1}),
        0.5, abs_tol=1e-05)
    assert math.isclose(profiler._calculate_net_cost(1.11, {'model_a': 1}),
        0.1, abs_tol=1e-05)
    assert math.isclose(profiler._calculate_net_cost(3.51, {'model_a': 2,
        'model_b': 1, 'model_c': 1, 'model_d': 1}), 0.2, abs_tol=1e-05)
