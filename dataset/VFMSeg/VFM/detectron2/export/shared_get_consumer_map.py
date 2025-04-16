def get_consumer_map(ssa):
    """
    Return dict from versioned blob to list of (i, j),
        where i is index of consumer op, j is the index of input of that op.
    """
    consumer_map = collections.defaultdict(list)
    for i in range(len(ssa)):
        inputs = ssa[i][0]
        for j, inp in enumerate(inputs):
            consumer_map[inp].append((i, j))
    return consumer_map
