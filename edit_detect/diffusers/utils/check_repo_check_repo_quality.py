def check_repo_quality():
    """Check all models are properly tested and documented."""
    print('Checking all models are included.')
    check_model_list()
    print('Checking all models are public.')
    check_models_are_in_init()
    print('Checking all models are properly tested.')
    check_all_decorator_order()
    check_all_models_are_tested()
    print('Checking all objects are properly documented.')
    check_all_objects_are_documented()
    print('Checking all models are in at least one auto class.')
    check_all_models_are_auto_configured()
