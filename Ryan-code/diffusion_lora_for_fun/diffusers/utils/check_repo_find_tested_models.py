def find_tested_models(test_file):
    """Parse the content of test_file to detect what's in all_model_classes"""
    with open(os.path.join(PATH_TO_TESTS, test_file), 'r', encoding='utf-8',
        newline='\n') as f:
        content = f.read()
    all_models = re.findall('all_model_classes\\s+=\\s+\\(\\s*\\(([^\\)]*)\\)',
        content)
    all_models += re.findall('all_model_classes\\s+=\\s+\\(([^\\)]*)\\)',
        content)
    if len(all_models) > 0:
        model_tested = []
        for entry in all_models:
            for line in entry.split(','):
                name = line.strip()
                if len(name) > 0:
                    model_tested.append(name)
        return model_tested
