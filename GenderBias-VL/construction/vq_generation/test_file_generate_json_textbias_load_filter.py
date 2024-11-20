def load_filter(test_case_file):
    with open(test_case_file, 'r') as f:
        filter_keys = f.readlines()
    filter_keys = [x.strip() for x in filter_keys]
    filter_keys = set(filter_keys)
    return filter_keys
