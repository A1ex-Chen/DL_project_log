def writer_unordered(output, q, n_workers):
    workers_done = 0
    n_items = 0
    with open(output, 'w') as f:
        while True:
            m = q.get()
            if m is None:
                workers_done += 1
                if workers_done == n_workers:
                    break
                continue
            _, res = m
            f.write(json.dumps(res) + '\n')
            n_items += 1
    return n_items
