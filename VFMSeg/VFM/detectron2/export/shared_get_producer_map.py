def get_producer_map(ssa):
    """
    Return dict from versioned blob to (i, j),
        where i is index of producer op, j is the index of output of that op.
    """
    producer_map = {}
    for i in range(len(ssa)):
        outputs = ssa[i][1]
        for j, outp in enumerate(outputs):
            producer_map[outp] = i, j
    return producer_map
