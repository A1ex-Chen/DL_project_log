@classmethod
def _iterate_test_cases(cls, model_key: Optional[str]=None):
    if not hasattr(cls, 'TEST_CASES') and not (model_key or any(
        'test_cases' in conf for conf in cls.CONFIGURATIONS.values())) and (
        model_key and 'test_cases' not in cls.CONFIGURATIONS[model_key]
        ) or model_key and model_key not in cls.CONFIGURATIONS:
        logger.debug('No test cases defined', model_type=cls.__name__)
        return
    model_keys = [model_key] if model_key else cls.CONFIGURATIONS.keys()
    cls_test_cases: List[Union[TestCase[ItemType, ReturnType], Dict]] = []
    if hasattr(cls, 'TEST_CASES'):
        cls_test_cases = cls.TEST_CASES
    for model_key in model_keys:
        for case in cls_test_cases:
            if isinstance(case, dict):
                case = TestCase(**case)
            yield model_key, case.item, case.result, case.keyword_args
        conf = cls.CONFIGURATIONS[model_key]
        if 'test_cases' not in conf:
            continue
        for case in conf['test_cases']:
            if isinstance(case, dict):
                case = TestCase(**case)
            yield model_key, case.item, case.result, case.keyword_args
