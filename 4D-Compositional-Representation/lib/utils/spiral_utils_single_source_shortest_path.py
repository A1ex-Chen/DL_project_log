def single_source_shortest_path(V, E, source, dist=None, prev=None):
    import heapq
    if dist == None:
        dist = [None for i in range(len(V))]
        prev = [None for i in range(len(V))]
    q = []
    seen = set()
    heapq.heappush(q, (0, source, None))
    while len(q) > 0 and len(seen) < len(V):
        d_, v, p = heapq.heappop(q)
        if v in seen:
            continue
        seen.add(v)
        prev[v] = p
        dist[v] = d_
        for w in E[v]:
            if w in seen:
                continue
            dw = d_ + distance(V[v], V[w])
            heapq.heappush(q, (dw, w, v))
    return prev, dist
