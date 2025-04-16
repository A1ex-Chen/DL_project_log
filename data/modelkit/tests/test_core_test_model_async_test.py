def test_model_async_test():


    class TestClass(AsyncModel):
        TEST_CASES = SYNC_ASYNC_TEST_CASES

        async def _predict(self, item, **_):
            await asyncio.sleep(0)
            return len(item)
    TestClass().test()
