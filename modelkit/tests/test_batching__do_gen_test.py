def _do_gen_test(m, batch_size, n_items):

    def item_iterator():
        yield from range(n_items)
    for value, position_in_batch, batch_len in m.predict_gen(item_iterator(
        ), batch_size=batch_size):
        assert position_in_batch == value % batch_size
        if value < n_items // batch_size * batch_size:
            assert batch_len == batch_size
        else:
            assert batch_len == n_items - n_items // batch_size * batch_size
