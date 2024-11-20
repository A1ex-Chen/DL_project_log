def parse_line(line):
    """
        Parse information from a line in a requirements text file
        """
    if line.startswith('-r '):
        target = line.split(' ')[1]
        for info in parse_require_file(target):
            yield info
    else:
        info = {'line': line}
        if line.startswith('-e '):
            info['package'] = line.split('#egg=')[1]
        else:
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]
            info['package'] = parts[0]
            if len(parts) > 1:
                op, rest = parts[1:]
                if ';' in rest:
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest
                info['version'] = op, version
        yield info
