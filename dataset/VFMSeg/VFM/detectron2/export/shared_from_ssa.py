@staticmethod
def from_ssa(ssa):
    graph = DiGraph()
    for op_id in range(len(ssa)):
        for inp in ssa[op_id][0]:
            for outp in ssa[op_id][1]:
                graph.add_edge(inp, outp)
    return graph
