def generate_spirals(step_sizes, M, Adj, Trigs, reference_points, dilation=
    None, random=False, meshpackage='mpi-mesh', counter_clockwise=True,
    nb_stds=2):
    Adj_spirals = []
    for i in range(len(Adj)):
        if meshpackage == 'trimesh':
            mesh_vertices = M[i].vertices
        elif meshpackage == 'mpi-mesh':
            mesh_vertices = M[i].v
        sp = get_spirals(mesh_vertices, Adj[i], Trigs[i], reference_points[
            i], n_steps=step_sizes[i], padding='zero', counter_clockwise=
            counter_clockwise, random=random)
        Adj_spirals.append(sp)
        print('spiral generation for hierarchy %d (%d vertices) finished' %
            (i, len(Adj_spirals[-1])))
    if dilation:
        for i in range(len(dilation)):
            dil = dilation[i]
            dil_spirals = []
            for j in range(len(Adj_spirals[i])):
                s = Adj_spirals[i][j][:1] + Adj_spirals[i][j][1::dil]
                dil_spirals.append(s)
            Adj_spirals[i] = dil_spirals
    L = []
    for i in range(len(Adj_spirals)):
        L.append([])
        for j in range(len(Adj_spirals[i])):
            L[i].append(len(Adj_spirals[i][j]))
        L[i] = np.array(L[i])
    spiral_sizes = []
    for i in range(len(L)):
        sz = L[i].mean() + nb_stds * L[i].std()
        spiral_sizes.append(int(sz))
        print('spiral sizes for hierarchy %d:  %d' % (i, spiral_sizes[-1]))
    spirals_np = []
    for i in range(len(spiral_sizes)):
        S = np.zeros((1, len(Adj_spirals[i]) + 1, spiral_sizes[i])) - 1
        for j in range(len(Adj_spirals[i])):
            S[0, j, :len(Adj_spirals[i][j])] = Adj_spirals[i][j][:
                spiral_sizes[i]]
        spirals_np.append(S)
    return spirals_np, spiral_sizes, Adj_spirals
