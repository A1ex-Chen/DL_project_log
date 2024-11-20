def _get_dependency_chain(ssa, versioned_target, versioned_source):
    """
    Return the index list of relevant operator to produce target blob from source blob,
        if there's no dependency, return empty list.
    """
    consumer_map = get_consumer_map(ssa)
    producer_map = get_producer_map(ssa)
    start_op = min(x[0] for x in consumer_map[versioned_source]) - 15
    end_op = producer_map[versioned_target][0
        ] + 15 if versioned_target in producer_map else start_op
    sub_graph_ssa = ssa[start_op:end_op + 1]
    if len(sub_graph_ssa) > 30:
        logger.warning(
            'Subgraph bebetween {} and {} is large (from op#{} to op#{}), it might take non-trival time to find all paths between them.'
            .format(versioned_source, versioned_target, start_op, end_op))
    dag = DiGraph.from_ssa(sub_graph_ssa)
    paths = dag.get_all_paths(versioned_source, versioned_target)
    ops_in_paths = [[producer_map[blob][0] for blob in path[1:]] for path in
        paths]
    return sorted(set().union(*[set(ops) for ops in ops_in_paths]))
