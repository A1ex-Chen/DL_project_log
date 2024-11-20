@pytest.mark.parametrize('items,batch_size,expected_steps,expected_steps_gen',
    CALLBACK_TEST_CASES)
def test_callback_batch_process(items, batch_size, expected_steps,
    expected_steps_gen):
    steps = 0

    def _callback(batch_step, batch_items, batch_results):
        nonlocal steps
        nonlocal items
        if batch_size:
            assert items[batch_step:batch_step + batch_size] == batch_items
        assert callback_func(batch_items) == batch_results
        steps += 1
    m = SomeModel()
    m.predict_batch(items, batch_size=batch_size, _callback=_callback)
    assert steps == expected_steps
    steps = 0

    def _callback_gen(batch_step, batch_items, batch_results):
        nonlocal steps
        steps += 1
        if batch_size:
            assert items[batch_step:batch_step + batch_size] == batch_items
        else:
            assert len(batch_items) == 1
        assert callback_func(batch_items) == batch_results
    list(m.predict_gen(iter(items), batch_size=batch_size, _callback=
        _callback_gen))
    assert steps == expected_steps_gen
