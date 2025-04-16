def test_method__compute_sub_calls_and_update_graph_calls():
    model_library = ModelLibrary(models=[ModelA, ModelB, ModelC, ModelD,
        Pipeline])
    pipeline = model_library.get('pipeline')
    item = {'abc': 123}
    profiler = SimpleProfiler(pipeline)
    _ = pipeline.predict(item)
    assert profiler._compute_sub_calls_and_update_graph_calls('pipeline', {}
        ) == {'model_a': 2, 'model_c': 1, 'model_b': 1, 'model_d': 1}
    assert profiler.graph_calls['pipeline']['__main__'] == 2
    assert profiler._compute_sub_calls_and_update_graph_calls('model_a', {}
        ) == {}
    assert profiler.graph_calls['model_a']['__main__'] == 3
    assert profiler._compute_sub_calls_and_update_graph_calls('model_b', {
        'model_a': 2}) == {'model_a': 1}
    assert profiler.graph_calls['model_b']['__main__'] == 2
    assert profiler._compute_sub_calls_and_update_graph_calls('model_c', {}
        ) == {}
    assert profiler.graph_calls['model_c']['__main__'] == 2
    assert profiler._compute_sub_calls_and_update_graph_calls('model_d', {
        'model_a': 2}) == {'model_a': 1}
    assert profiler.graph_calls['model_d']['__main__'] == 2
