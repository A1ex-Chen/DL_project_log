def _convert_cost_matrix_to_graph(self, cost_matrices: list[np.ndarray],
    no_detection_cost: float=100000.0) ->tuple[list[int], list[int], list[
    int], list[int], list[int], dict[int, Node], int, int]:
    """
        Converts cost matrix to graph representation for optimization.

        Args:
            cost_matrices: List of Numpy arrays representing the cost matrices.
            no_detection_cost: Cost to be used when there is no detection.

        Returns:
            start_nodes: List of start nodes.
            end_nodes: List of end nodes.
            capacities: List of the capacities of the arcs.
            unit_costs: List of the unit costs of the arcs.
            supplies: List of the supplies for the nodes.
            node_to_detection: Dictionary mapping node to a Node namedtuple.
            source_node: Source node index.
            sink_node: Sink node index.
        """
    G = nx.DiGraph()
    frame_to_nodes: DefaultDict[int, list[int]] = defaultdict(list)
    num_frames = len(cost_matrices)
    source_node = 0
    sink_node = 10 ** 9
    G.add_node(source_node, demand=-1, frame=-1)
    G.add_node(sink_node, demand=1, frame=num_frames)
    curr_node = 1
    for frame in range(num_frames):
        num_detections_curr_frame = cost_matrices[frame].shape[1] + 1
        num_detections_prev_frame = len(frame_to_nodes[frame - 1])
        for detection_curr in range(num_detections_curr_frame):
            is_dummy_node = detection_curr + 1 == num_detections_curr_frame
            if frame == 0:
                cost = no_detection_cost if is_dummy_node else cost_matrices[0
                    ][0][detection_curr]
                G.add_node(curr_node, demand=0, frame=frame, detection=
                    detection_curr, is_dummy=is_dummy_node)
                G.add_edge(source_node, curr_node, capacity=1, weight=cost)
                if curr_node not in frame_to_nodes[frame]:
                    frame_to_nodes[frame].append(curr_node)
            else:
                for detection_prev in range(num_detections_prev_frame):
                    is_dummy_node_prev = (detection_prev + 1 ==
                        num_detections_prev_frame)
                    no_detection = (is_dummy_node or is_dummy_node_prev or 
                        detection_prev >= num_detections_prev_frame or 
                        detection_curr >= num_detections_curr_frame)
                    cost = (no_detection_cost if no_detection else
                        cost_matrices[frame][detection_prev][detection_curr])
                    prev_node = frame_to_nodes[frame - 1][detection_prev]
                    G.add_node(curr_node, demand=0, frame=frame, detection=
                        detection_curr, is_dummy=is_dummy_node)
                    G.add_edge(prev_node, curr_node, capacity=1, weight=cost)
                    if curr_node not in frame_to_nodes[frame]:
                        frame_to_nodes[frame].append(curr_node)
            curr_node += 1
        if frame == num_frames - 1:
            for node in frame_to_nodes[frame]:
                G.add_edge(node, sink_node, capacity=1, weight=0)
    return G
