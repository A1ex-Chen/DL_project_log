def _do_model_test(model, ITEMS):
    for i in ITEMS:
        res = model(i, _force_compute=True)
        if isinstance(res, pydantic.BaseModel):
            res = res.model_dump()
        assert i == res
    batch_results = model.predict_batch(ITEMS)
    if isinstance(batch_results[0], pydantic.BaseModel):
        batch_results = [res.model_dump() for res in batch_results]
    assert batch_results == ITEMS
    batch_results = model.predict_batch(ITEMS + [{'ok': {'boomer': [-1]}}])
    if isinstance(batch_results[0], pydantic.BaseModel):
        batch_results = [res.model_dump() for res in batch_results]
    assert batch_results == ITEMS + [{'ok': {'boomer': [-1]}}]
