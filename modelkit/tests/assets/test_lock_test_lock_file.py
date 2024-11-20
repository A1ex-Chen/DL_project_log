def test_lock_file(working_dir):
    lock_path = os.path.join(working_dir, 'lock')
    threads = []
    for _ in range(3):
        t = _start_wait_process(lock_path, 2)
        threads.append(t)
    ranges = []
    while threads:
        t = threads.pop()
        res = t()
        assert res is not None
        lines = res.splitlines()
        assert len(lines) == 2
        start = lines[0]
        end = lines[1]
        ranges.append((float(start), float(end)))
    ranges.sort()
    for i in range(len(ranges) - 1):
        end = ranges[i][1]
        start = ranges[i + 1][0]
        assert end <= start
