def test_generator_exit_callback():
    steps = 0
    batch_size = 32
    batch_break = int(batch_size / 2)

    def _callback(batch_step, batch_items, batch_results):
        nonlocal steps
        assert len(batch_items) == batch_size
        assert len(batch_results) == batch_break
        assert callback_func(batch_items)[:batch_break] == batch_results
        assert batch_step < 1
        steps += 1
    m = SomeModel()
    for i, _prediction in enumerate(m.predict_gen(range(100), _callback=
        _callback, batch_size=batch_size)):
        if i + 1 == batch_break:
            break
    assert steps == 1
