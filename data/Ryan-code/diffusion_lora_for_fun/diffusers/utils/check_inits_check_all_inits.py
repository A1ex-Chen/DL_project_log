def check_all_inits():
    """
    Check all inits in the transformers repo and raise an error if at least one does not define the same objects in
    both halves.
    """
    failures = []
    for root, _, files in os.walk(PATH_TO_TRANSFORMERS):
        if '__init__.py' in files:
            fname = os.path.join(root, '__init__.py')
            objects = parse_init(fname)
            if objects is not None:
                errors = analyze_results(*objects)
                if len(errors) > 0:
                    errors[0] = f"""Problem in {fname}, both halves do not define the same objects.
{errors[0]}"""
                    failures.append('\n'.join(errors))
    if len(failures) > 0:
        raise ValueError('\n\n'.join(failures))
