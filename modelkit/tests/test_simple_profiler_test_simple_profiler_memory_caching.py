def test_simple_profiler_memory_caching():
    """A test case simulating memory caching for SimpleProfiler"""
    model_library = ModelLibrary(models=[ModelA, ModelB, ModelC, ModelD,
        Pipeline2])
    pipeline = model_library.get('pipeline2')
    item = {'abc': 123}
    profiler = SimpleProfiler(pipeline)
    _ = pipeline.predict(item)
    stat = profiler.summary()
    assert set(stat['Name']) == set(['pipeline2', 'model_b', 'model_a',
        'model_c', 'model_d'])
    graph = profiler.graph
    assert graph['model_a'] == set()
    assert graph['model_b'] == set(['model_a'])
    assert graph['model_c'] == set()
    assert graph['model_d'] == set(['model_a'])
    assert graph['pipeline2'] == set(['model_a', 'model_b', 'model_c',
        'model_d'])
    graph_calls = profiler.graph_calls
    assert graph_calls['model_a'] == {'__main__': 3}
    assert graph_calls['model_b'] == {'__main__': 1, 'model_a': 1}
    assert graph_calls['model_c'] == {'__main__': 1}
    assert graph_calls['model_d'] == {'__main__': 1, 'model_a': 1}
    assert graph_calls['pipeline2'] == {'__main__': 1, 'model_a': 3,
        'model_b': 1, 'model_c': 1, 'model_d': 1}
    num_call = dict(zip(stat['Name'], stat['Num call']))
    assert num_call == {'pipeline2': 1, 'model_a': 3, 'model_b': 1,
        'model_d': 1, 'model_c': 1}
    assert math.isclose(sum(stat['Net percentage %']), 100, abs_tol=1.0)
    assert math.isclose(stat['Total percentage %'][0], 99.9, abs_tol=1.0)
    table_str = profiler.summary(print_table=True)
    assert isinstance(table_str, str)
