def test_simple_profiler_dynamic_graph():
    """A more complicated test case with dynamic model.
    A dynamic model has different duration depending on its arguments.
    The dynamic model is used to simulate "caching" in model "predict".
    """
    model_library = ModelLibrary(models=[ModelX, ModelY, ModelZ,
        DynamicModel, DynamicPipeline])
    pipeline = model_library.get('dynamic_pipeline')
    item = {'abc': 123}
    profiler = SimpleProfiler(pipeline)
    _ = pipeline.predict(item)
    stat = profiler.summary()
    assert set(stat['Name']) == set(['model_x', 'model_y', 'model_z',
        'dynamic_model', 'dynamic_pipeline'])
    graph = profiler.graph
    assert graph['model_x'] == set()
    assert graph['model_y'] == set(['model_x'])
    assert graph['model_z'] == set()
    assert graph['dynamic_model'] == set(['model_x', 'model_y', 'model_z'])
    assert graph['dynamic_pipeline'] == set(['dynamic_model'])
    graph_calls = profiler.graph_calls
    assert graph_calls['model_x'] == {'__main__': 2}
    assert graph_calls['model_y'] == {'__main__': 1, 'model_x': 1}
    assert graph_calls['model_z'] == {'__main__': 2}
    assert graph_calls['dynamic_model'] == {'__main__': 2, 'model_x': 2,
        'model_y': 1, 'model_z': 2}
    assert graph_calls['dynamic_pipeline'] == {'__main__': 1,
        'dynamic_model': 2, 'model_x': 2, 'model_y': 1, 'model_z': 2}
    num_call = dict(zip(stat['Name'], stat['Num call']))
    assert num_call == {'dynamic_pipeline': 1, 'dynamic_model': 2,
        'model_x': 2, 'model_y': 1, 'model_z': 2}
    assert math.isclose(sum(stat['Net percentage %']), 100, abs_tol=1.0)
    assert math.isclose(stat['Total percentage %'][0], 99.9, abs_tol=1.0)
    table_str = profiler.summary(print_table=True)
    assert isinstance(table_str, str)
