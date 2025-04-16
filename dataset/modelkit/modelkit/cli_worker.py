def worker(lib, model_name, q_in, q):
    model = lib.get(model_name)
    n = 0
    done = False
    while not done:
        items = []
        indices = []
        while True:
            m = q_in.get()
            if m is None:
                done = True
                break
            k, item = m
            items.append(item)
            indices.append(k)
            if model.batch_size is None or len(items) >= model.batch_size:
                break
        for k, res in zip(indices, model.predict_gen(items)):
            q.put((k, res))
            n += 1
    q.put(None)
    return n
