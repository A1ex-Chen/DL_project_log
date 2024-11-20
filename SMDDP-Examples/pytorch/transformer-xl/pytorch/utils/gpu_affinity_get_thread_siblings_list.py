def get_thread_siblings_list():
    path = '/sys/devices/system/cpu/cpu*/topology/thread_siblings_list'
    thread_siblings_list = []
    pattern = re.compile('(\\d+)\\D(\\d+)')
    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname) as f:
            content = f.read().strip()
            res = pattern.findall(content)
            if res:
                pair = tuple(map(int, res[0]))
                thread_siblings_list.append(pair)
    return thread_siblings_list
