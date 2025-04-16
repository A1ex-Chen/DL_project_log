def _compare_models(model0, model1, items_and_results, tolerance=0.01):
    """compares two models in the following situations:
    - model0 per item vs. model1 per item
    - model0 batched vs. model1 batched
    - model0 per item vs. model0 batched
    """
    res_model0_per_item = []
    try:
        for item, result in items_and_results:
            res_model0 = model0.predict(item)
            res_model0_per_item.append(res_model0)
            res_model1 = model1.predict(item)
            assert compare_result(res_model0, result, tolerance)
            assert compare_result(res_model0, res_model1, tolerance)
    except AssertionError as e:
        raise AssertionError(f'Models differ on single items\n{e.args[0]}'
            ) from e
    items = [item for item, _ in items_and_results]
    try:
        res_model0_items = model0.predict_batch(items)
        res_model1_items = model1.predict_batch(items)
        for k in range(len(items)):
            res_model0 = res_model0_items[k]
            res_model1 = res_model1_items[k]
            assert compare_result(res_model0, res_model1, tolerance)
    except AssertionError as e:
        raise AssertionError(f'Models differ on item batches\n{e.args[0]}'
            ) from e
    try:
        for k in range(len(items)):
            assert compare_result(res_model0_items[k], res_model0_per_item[
                k], tolerance)
    except AssertionError as e:
        raise AssertionError(
            f'Models predictions on single and batches differ\n{e.args[0]}'
            ) from e
