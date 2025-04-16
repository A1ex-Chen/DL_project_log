def overwrite_file(file, class_name, test_name, correct_line, done_test):
    _id = f'{file}_{class_name}_{test_name}'
    done_test[_id] += 1
    with open(file, 'r') as f:
        lines = f.readlines()
    class_regex = f'class {class_name}('
    test_regex = f"{4 * ' '}def {test_name}("
    line_begin_regex = f"{8 * ' '}{correct_line.split()[0]}"
    another_line_begin_regex = f"{16 * ' '}{correct_line.split()[0]}"
    in_class = False
    in_func = False
    in_line = False
    insert_line = False
    count = 0
    spaces = 0
    new_lines = []
    for line in lines:
        if line.startswith(class_regex):
            in_class = True
        elif in_class and line.startswith(test_regex):
            in_func = True
        elif in_class and in_func and (line.startswith(line_begin_regex) or
            line.startswith(another_line_begin_regex)):
            spaces = len(line.split(correct_line.split()[0])[0])
            count += 1
            if count == done_test[_id]:
                in_line = True
        if in_class and in_func and in_line:
            if ')' not in line:
                continue
            else:
                insert_line = True
        if in_class and in_func and in_line and insert_line:
            new_lines.append(f"{spaces * ' '}{correct_line}")
            in_class = in_func = in_line = insert_line = False
        else:
            new_lines.append(line)
    with open(file, 'w') as f:
        for line in new_lines:
            f.write(line)
