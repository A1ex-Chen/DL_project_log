def test(achieved, target, name, higher_better=True):
    passed = True
    if target is not None and achieved is not None:
        logging.info(f'{name} achieved: {achieved:.2f} target: {target:.2f}')
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
