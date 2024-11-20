def get_spirals(mesh, adj, trig, reference_points, n_steps=1, padding=
    'zero', counter_clockwise=True, random=False):
    spirals = []
    if not random:
        heat_path = None
        dist = None
        for reference_point in reference_points:
            heat_path, dist = single_source_shortest_path(mesh, adj,
                reference_point, dist, heat_path)
        heat_source = reference_points
    for i in range(mesh.shape[0]):
        seen = set()
        seen.add(i)
        trig_central = list(trig[i])
        A = adj[i]
        spiral = [i]
        if not random:
            if i in heat_source:
                shortest_dist = np.inf
                init_vert = None
                for neighbor in A:
                    d = np.sum(np.square(mesh[i] - mesh[neighbor]))
                    if d < shortest_dist:
                        shortest_dist = d
                        init_vert = neighbor
            else:
                init_vert = heat_path[i]
        else:
            init_vert = choice(A)
        if init_vert is not None:
            ring = [init_vert]
            seen.add(init_vert)
        else:
            ring = []
        while len(trig_central) > 0 and init_vert is not None:
            cur_v = ring[-1]
            cur_t = [t for t in trig_central if t in trig[cur_v]]
            if len(ring) == 1:
                orientation_0 = cur_t[0][0] == i and cur_t[0][1
                    ] == cur_v or cur_t[0][1] == i and cur_t[0][2
                    ] == cur_v or cur_t[0][2] == i and cur_t[0][0] == cur_v
                if not counter_clockwise:
                    orientation_0 = not orientation_0
                if len(cur_t) >= 2:
                    if orientation_0:
                        third = [p for p in cur_t[0] if p != i and p != cur_v][
                            0]
                        trig_central.remove(cur_t[0])
                    else:
                        third = [p for p in cur_t[1] if p != i and p != cur_v][
                            0]
                        trig_central.remove(cur_t[1])
                    ring.append(third)
                    seen.add(third)
                elif len(cur_t) == 1:
                    break
            elif len(cur_t) >= 1:
                third = [p for p in cur_t[0] if p != cur_v and p != i][0]
                if third not in seen:
                    ring.append(third)
                    seen.add(third)
                trig_central.remove(cur_t[0])
            elif len(cur_t) == 0:
                break
        rev_i = len(ring)
        if init_vert is not None:
            v = init_vert
            if orientation_0 and len(ring) == 1:
                reverse_order = False
            else:
                reverse_order = True
        need_padding = False
        while len(trig_central) > 0 and init_vert is not None:
            cur_t = [t for t in trig_central if t in trig[v]]
            if len(cur_t) != 1:
                break
            else:
                need_padding = True
            third = [p for p in cur_t[0] if p != v and p != i][0]
            trig_central.remove(cur_t[0])
            if third not in seen:
                ring.insert(rev_i, third)
                seen.add(third)
                if not reverse_order:
                    rev_i = len(ring)
                v = third
        if need_padding:
            ring.insert(rev_i, -1)
            """
            ring_copy = list(ring[1:])
            rev_i = rev_i - 1
            for z in range(len(ring_copy)-2):
                if padding == 'zero':
                    ring.insert(rev_i,-1) # -1 is our sink node
                elif padding == 'mirror':
                    ring.insert(rev_i,ring_copy[rev_i-z-1])
            """
        spiral += ring
        for step in range(n_steps - 1):
            next_ring = set([])
            next_trigs = set([])
            if len(ring) == 0:
                break
            base_triangle = None
            init_vert = None
            for w in ring:
                if w != -1:
                    for u in adj[w]:
                        if u not in seen:
                            next_ring.add(u)
            for u in next_ring:
                for tr in trig[u]:
                    if len([x for x in tr if x in seen]) == 1:
                        next_trigs.add(tr)
                    elif ring[0] in tr and ring[-1] in tr:
                        base_triangle = tr
            if base_triangle is not None:
                init_vert = [x for x in base_triangle if x != ring[0] and x !=
                    ring[-1]]
                if len(list(next_trigs.intersection(set(trig[init_vert[0]])))
                    ) == 0:
                    init_vert = None
            if init_vert is None:
                for r in range(len(ring) - 1):
                    if ring[r] != -1 and ring[r + 1] != -1:
                        tr = [t for t in trig[ring[r]] if t in trig[ring[r +
                            1]]]
                        for t in tr:
                            init_vert = [v for v in t if v not in seen]
                            if len(init_vert) > 0 and len(list(next_trigs.
                                intersection(set(trig[init_vert[0]])))) > 0:
                                break
                            else:
                                init_vert = []
                        if len(init_vert) > 0 and len(list(next_trigs.
                            intersection(set(trig[init_vert[0]])))) > 0:
                            break
                        else:
                            init_vert = []
            if init_vert is None:
                init_vert = []
            if len(init_vert) > 0:
                init_vert = init_vert[0]
                ring = [init_vert]
                seen.add(init_vert)
            else:
                init_vert = None
                ring = []
            while len(next_trigs) > 0 and init_vert is not None:
                cur_v = ring[-1]
                cur_t = list(next_trigs.intersection(set(trig[cur_v])))
                if len(ring) == 1:
                    try:
                        orientation_0 = cur_t[0][0] in seen and cur_t[0][1
                            ] == cur_v or cur_t[0][1] in seen and cur_t[0][2
                            ] == cur_v or cur_t[0][2] in seen and cur_t[0][0
                            ] == cur_v
                    except:
                        import pdb
                        pdb.set_trace()
                    if not counter_clockwise:
                        orientation_0 = not orientation_0
                    if len(cur_t) >= 2:
                        if orientation_0:
                            third = [p for p in cur_t[0] if p not in seen and
                                p != cur_v][0]
                            next_trigs.remove(cur_t[0])
                        else:
                            third = [p for p in cur_t[1] if p not in seen and
                                p != cur_v][0]
                            next_trigs.remove(cur_t[1])
                        ring.append(third)
                        seen.add(third)
                    elif len(cur_t) == 1:
                        break
                elif len(cur_t) >= 1:
                    third = [p for p in cur_t[0] if p != v and p not in seen]
                    next_trigs.remove(cur_t[0])
                    if len(third) > 0:
                        third = third[0]
                        if third not in seen:
                            ring.append(third)
                            seen.add(third)
                    else:
                        break
                elif len(cur_t) == 0:
                    break
            rev_i = len(ring)
            if init_vert is not None:
                v = init_vert
                if orientation_0 and len(ring) == 1:
                    reverse_order = False
                else:
                    reverse_order = True
            need_padding = False
            while len(next_trigs) > 0 and init_vert is not None:
                cur_t = [t for t in next_trigs if t in trig[v]]
                if len(cur_t) != 1:
                    break
                else:
                    need_padding = True
                third = [p for p in cur_t[0] if p != v and p not in seen]
                next_trigs.remove(cur_t[0])
                if len(third) > 0:
                    third = third[0]
                    if third not in seen:
                        ring.insert(rev_i, third)
                        seen.add(third)
                    if not reverse_order:
                        rev_i = len(ring)
                    v = third
            if need_padding:
                ring.insert(rev_i, -1)
                """
                ring_copy = list(ring[1:])
                rev_i = rev_i - 1
                for z in range(len(ring_copy)-2):
                    if padding == 'zero':
                        ring.insert(rev_i,-1) # -1 is our sink node
                    elif padding == 'mirror':
                        ring.insert(rev_i,ring_copy[rev_i-z-1])
                """
            spiral += ring
        spirals.append(spiral)
    return spirals
