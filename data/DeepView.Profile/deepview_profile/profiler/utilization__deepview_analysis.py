def _deepview_analysis(self, filepath):
    startTime = time.time()
    tp = self._get_perfetto_object(filepath)
    profilerStepStart = tp.query_dict(
        "select * from slices where name like '%ProfilerStep%'")
    main_track = profilerStepStart[0]['track_id']
    start = profilerStepStart[0]['ts']
    end = start + profilerStepStart[0]['dur']
    profilerStartDepth = profilerStepStart[0]['depth']
    rootQuery = tp.query_dict(
        f"""
                                select * from slices where name like '%nn.Module:%' 
                                and depth = 
                                (SELECT MIN(depth) from slices where name like '%nn.Module%' and depth>{profilerStartDepth} and ts between {start} and {end} and track_id = {main_track})                
                                """
        )[0]
    rootNode = Node(rootQuery['name'], rootQuery['ts'], rootQuery['ts'] +
        rootQuery['dur'], rootQuery['dur'], rootQuery['track_id'],
        rootQuery['depth'], rootQuery['slice_id'], rootQuery['dur'], 0, 0, 0)
    self._calculate_gpu_forward_time(tp, rootNode)
    logger.debug(f'{rootNode}\n')
    stack = deque([rootNode])
    while stack:
        node = stack.popleft()
        queryNewDepth = tp.query_dict(
            f"""select MIN(depth) from slices where (name like '%nn.Module:%' or name like '%aten::%') and 
                                        depth>{node.depth} and track_id={node.track}
                                        and ts between {node.start} and {node.end}"""
            )
        minDepth = queryNewDepth[0]['min(depth)'] if queryNewDepth else -1
        if minDepth:
            queryResults = tp.query_dict(
                f"""select * from slices where (name like '%nn.Module:%' or name like '%aten::%') 
                                        and depth={minDepth} and track_id={node.track} 
                                            and ts between {node.start} and {node.end}"""
                )
            for qr in queryResults:
                newNode = Node(qr['name'], qr['ts'], qr['ts'] + qr['dur'],
                    qr['dur'], qr['track_id'], qr['depth'], qr['slice_id'],
                    qr['dur'], 0, 0, 0)
                self._calculate_gpu_forward_time(tp, newNode)
                node.children.append(newNode)
                stack.append(newNode)
    ForwardPostOrderTraversal = list(filter(lambda x: 'aten::' in x.name,
        rootNode.postorder()))
    ForwardPostOrderTraversal.sort(key=lambda x: x.start, reverse=True)
    backwardSlices = self._backward_slices(tp)
    matchings = self._lcs(ForwardPostOrderTraversal, backwardSlices)
    countAccumulateGrad = 0
    for slice in backwardSlices:
        if 'AccumulateGrad' in slice['name']:
            countAccumulateGrad += 1
    numValidBackwardSlices = len(backwardSlices) - countAccumulateGrad
    if logging.root.level == logging.DEBUG:
        logger.debug(f'number of valid slices: {numValidBackwardSlices}\n')
        logger.debug(
            f"""Number of matched slices: {len(matchings)} percentage: {round(len(matchings) / numValidBackwardSlices * 100, 2)}
"""
            )
    for match in matchings:
        node = match[0]
        backward_slice = match[1]
        node.cpu_backward_slices.append(backward_slice)
    if logging.root.level == logging.DEBUG:
        matchings.sort(key=lambda x: x[1]['ts'])
        with open('matching_results.txt', 'w') as file:
            file.write(
                f'Total number of backward slices: {len(backwardSlices)}\n')
            file.write(f'number of valid slices: {numValidBackwardSlices}\n')
            file.write(
                f"""Number of matched slices: {len(matchings)} percentage: {round(len(matchings) / numValidBackwardSlices * 100, 2)}
"""
                )
            for m in matchings:
                file.write(
                    f"""forward: {m[0].name} {m[0].slice_id} backward: {m[1]['name']} {m[1]['slice_id']}
"""
                    )
    self._accumulate_backward_slices_to_node(rootNode)
    self._populate_backward_data(rootNode)
    if logging.root.level == logging.DEBUG:
        profilingResult = self._convert_node_to_dict(rootNode)
        with open('profiling_results.json', 'wb') as f:
            f.write(orjson.dumps(profilingResult))
    endTime = time.time()
    logger.debug(f'Total elapsed time: {endTime - startTime}')
    self._root_node = rootNode
    self._tensor_core_perc = self._calculate_tensor_core_utilization(filepath)
    if not logging.root.level == logging.DEBUG:
        subprocess.run(['rm', '-f', os.path.join(os.getcwd(), filepath)])
