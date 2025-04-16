def run_tests(precision):
    dummy = TestPyProfNvtx('test_affine_grid', None)
    test_cases = list(filter(lambda x: 'test_' in x, map(lambda x: x[0],
        inspect.getmembers(dummy, predicate=inspect.ismethod))))
    print('Running tests for {}'.format(precision))
    suite = unittest.TestSuite()
    for test_case in test_cases:
        suite.addTest(TestPyProfNvtx(test_case, precision))
    unittest.TextTestRunner().run(suite)
