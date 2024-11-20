def writer(output, q, n_workers):
    next_index = 0
    items_to_write = []
    workers_done = 0
    done = False
    with open(output, 'w') as f:
        while not done:
            while True:
                m = q.get()
                if m is None:
                    workers_done += 1
                    if workers_done == n_workers:
                        done = True
                        break
                    continue
                k, res = m
                bisect.insort(items_to_write, (k, res))
                if k == next_index:
                    break
            while len(items_to_write) and items_to_write[0][0] == next_index:
                _, res = items_to_write.pop(0)
                f.write(json.dumps(res) + '\n')
                next_index += 1
    return next_index
