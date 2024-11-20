def fetch_pipeline_modules_to_test():
    try:
        pipeline_objects = fetch_pipeline_objects()
    except Exception as e:
        logger.error(e)
        raise RuntimeError('Unable to fetch model list from HuggingFace Hub.')
    test_modules = []
    for pipeline_name in pipeline_objects:
        module = getattr(diffusers, pipeline_name)
        test_module = module.__module__.split('.')[-2].strip()
        test_modules.append(test_module)
    return test_modules
