@pytest.mark.parametrize('fn, raises', TEST_CASES)
def test_typing_model_predict(fn, raises):
    _mypy_check_file(fn, raises)
