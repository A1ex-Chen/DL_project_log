def test_testing_model_library(testing_model_library):
    m = testing_model_library.get('some_model')
    assert m({'x': 1}).x == 1
