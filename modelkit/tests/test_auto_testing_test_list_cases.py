def test_list_cases():
    expected = [('some_model', {'x': 1}, {'x': 1}, {})]


    class SomeModel(Model[ModelItemType, ModelItemType]):
        CONFIGURATIONS = {'some_model': {}}
        TEST_CASES = [{'item': {'x': 1}, 'result': {'x': 1}}]

        def _predict(self, item):
            return item
    assert list(SomeModel._iterate_test_cases()) == expected
    assert list(SomeModel._iterate_test_cases('some_model')) == expected
    assert list(SomeModel._iterate_test_cases('unknown_model')) == []


    class TestableModel(Model[ModelItemType, ModelItemType]):
        CONFIGURATIONS = {'some_model': {}}
        TEST_CASES = [{'item': {'x': 1}, 'result': {'x': 1}}]

        def _predict(self, item):
            return item
    assert list(TestableModel._iterate_test_cases()) == expected
    assert list(TestableModel._iterate_test_cases('some_model')) == expected
    assert list(TestableModel._iterate_test_cases('unknown_model')) == []


    class TestableModel(Model[ModelItemType, ModelItemType]):
        CONFIGURATIONS = {'some_model': {'test_cases': [{'item': {'x': 1},
            'result': {'x': 1}}]}, 'some_other_model': {}}
        TEST_CASES = [{'item': {'x': 1}, 'result': {'x': 1}}]

        def _predict(self, item):
            return item
    assert list(TestableModel._iterate_test_cases()) == expected * 2 + [(
        'some_other_model', {'x': 1}, {'x': 1}, {})]
    assert list(TestableModel._iterate_test_cases('some_model')
        ) == expected * 2
    assert list(TestableModel._iterate_test_cases('unknown_model')) == []
    assert list(TestableModel._iterate_test_cases('some_other_model')) == [(
        'some_other_model', {'x': 1}, {'x': 1}, {})]
