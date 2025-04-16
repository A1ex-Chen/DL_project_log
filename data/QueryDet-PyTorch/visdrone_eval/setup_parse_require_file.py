def parse_require_file(fpath):
    with open(fpath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line and not line.startswith('#'):
                for info in parse_line(line):
                    yield info
