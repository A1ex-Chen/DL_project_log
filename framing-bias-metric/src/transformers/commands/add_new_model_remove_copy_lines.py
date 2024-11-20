def remove_copy_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    with open(path, 'w') as f:
        for line in lines:
            if '# Copied from transformers.' not in line:
                f.write(line)
