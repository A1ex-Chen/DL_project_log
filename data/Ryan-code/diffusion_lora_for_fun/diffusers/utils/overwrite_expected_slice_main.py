def main(correct, fail=None):
    if fail is not None:
        with open(fail, 'r') as f:
            test_failures = {l.strip() for l in f.readlines()}
    else:
        test_failures = None
    with open(correct, 'r') as f:
        correct_lines = f.readlines()
    done_tests = defaultdict(int)
    for line in correct_lines:
        file, class_name, test_name, correct_line = line.split(';')
        if test_failures is None or '::'.join([file, class_name, test_name]
            ) in test_failures:
            overwrite_file(file, class_name, test_name, correct_line,
                done_tests)
