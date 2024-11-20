def assert_param_count(model_1, model_2):
    count_1 = sum(p.numel() for p in model_1.parameters())
    count_2 = sum(p.numel() for p in model_2.parameters())
    assert count_1 == count_2, f'{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}'
