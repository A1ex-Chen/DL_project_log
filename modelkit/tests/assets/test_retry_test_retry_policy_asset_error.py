def test_retry_policy_asset_error():
    SHORT_RETRY_POLICY = copy.deepcopy(retry_policy(None))
    SHORT_RETRY_POLICY['stop'] = stop_after_attempt(2)
    k = 0

    @retry(**SHORT_RETRY_POLICY)
    def some_function():
        nonlocal k
        k += 1
        raise ObjectDoesNotExistError('driver', 'bucket', 'object')
    with pytest.raises(ObjectDoesNotExistError):
        some_function()
    assert k == 1
