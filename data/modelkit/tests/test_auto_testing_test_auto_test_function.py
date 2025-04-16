@pytest.mark.parametrize('test_case, do_raise', [({'item': True, 'result': 
    True}, False), ({'item': False, 'result': True}, True), ({'item': False,
    'keyword_args': {'force_true': True}, 'result': True}, False), ({'item':
    False, 'keyword_args': {'force_true': False}, 'result': True}, True)])
def test_auto_test_function(test_case, do_raise):


    class MyModel(Model):
        CONFIGURATIONS = {'my_model': {}}
        TEST_CASES = [test_case]

        def _predict(self, item: bool, force_true=False, **_) ->bool:
            if force_true:
                return True
            return item
    if do_raise:
        with pytest.raises(AssertionError):
            MyModel().test()
    else:
        MyModel().test()
