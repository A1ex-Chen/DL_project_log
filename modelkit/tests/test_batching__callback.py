def _callback(batch_step, batch_items, batch_results):
    nonlocal steps
    assert len(batch_items) == batch_size
    assert len(batch_results) == batch_break
    assert callback_func(batch_items)[:batch_break] == batch_results
    assert batch_step < 1
    steps += 1
