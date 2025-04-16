def test_model_sync_test():


    class TestClass(Model):
        TEST_CASES = SYNC_ASYNC_TEST_CASES

        def _predict(self, item, **_):
            return len(item)
    TestClass().test()
