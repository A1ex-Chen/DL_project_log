def benchmark(test_perplexity=None, target_perplexity=None, test_throughput
    =None, target_throughput=None):

    def test(achieved, target, name, higher_better=True):
        passed = True
        if target is not None and achieved is not None:
            logging.info(
                f'{name} achieved: {achieved:.2f} target: {target:.2f}')
            if higher_better:
                result = achieved >= target
            else:
                result = achieved <= target
            if result:
                logging.info(f'{name} test passed')
            else:
                logging.info(f'{name} test failed')
                passed = False
        return passed
    passed = True
    passed &= test(test_perplexity, target_perplexity, 'Perplexity', False)
    passed &= test(test_throughput, target_throughput, 'Throughput')
    return passed
