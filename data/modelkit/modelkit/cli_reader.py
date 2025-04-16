def reader(input, queues):
    queues_cycle = itertools.cycle(queues)
    q_in = next(queues_cycle)
    with open(input) as f:
        for k, l in enumerate(f):
            q_in.put((k, json.loads(l.strip())))
            q_in = next(queues_cycle)
    for q in queues:
        q.put(None)
