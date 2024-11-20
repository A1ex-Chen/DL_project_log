def _callback_gen(batch_step, batch_items, batch_results):
    nonlocal steps
    steps += 1
    if batch_size:
        assert items[batch_step:batch_step + batch_size] == batch_items
    else:
        assert len(batch_items) == 1
    assert callback_func(batch_items) == batch_results
