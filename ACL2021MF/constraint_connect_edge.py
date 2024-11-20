def connect_edge(self, M, additional_state, from_state, to_state, constraint):
    queue = [(from_state, c) for c in constraint]
    new_queue = []
    index2state = {}
    while len(queue) > 0:
        f_state, c = queue.pop(0)
        if len(c) == 1:
            M.add_connect(f_state, to_state, c)
        else:
            if c[0] not in index2state:
                index2state[c[0]] = additional_state
                additional_state += 1
            M.add_connect(f_state, index2state[c[0]], [c[0]])
            if not f_state == from_state:
                M.add_connect_except(f_state, from_state, [c[0]])
            new_queue.append((index2state[c[0]], c[1:]))
        if len(queue) == 0 and len(new_queue) > 0:
            queue = new_queue
            new_queue = []
            index2state = {}
    return M, additional_state
