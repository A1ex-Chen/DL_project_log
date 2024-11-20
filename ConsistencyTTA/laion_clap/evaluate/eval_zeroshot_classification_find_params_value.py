def find_params_value(file, key):
    with open(file, 'r') as f:
        for line in f:
            if key + ': ' in line:
                return line.split(': ')[1].strip()
    return None
