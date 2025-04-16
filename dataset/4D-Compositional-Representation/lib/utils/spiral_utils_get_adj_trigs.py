def get_adj_trigs(A, F, reference_mesh, meshpackage='mpi-mesh'):
    Adj = []
    for x in A:
        adj_x = []
        dx = x.todense()
        for i in range(x.shape[0]):
            adj_x.append(dx[i].nonzero()[1])
        Adj.append(adj_x)
    if meshpackage == 'trimesh':
        mesh_faces = reference_mesh.faces
    elif meshpackage == 'mpi-mesh':
        mesh_faces = reference_mesh.f
    trigs_full = [[] for i in range(len(Adj[0]))]
    for t in mesh_faces:
        u, v, w = t
        trigs_full[u].append((u, v, w))
        trigs_full[v].append((u, v, w))
        trigs_full[w].append((u, v, w))
    Trigs = [trigs_full]
    for i, T in enumerate(F):
        trigs_down = [[] for i in range(len(Adj[i + 1]))]
        for u, v, w in T:
            trigs_down[u].append((u, v, w))
            trigs_down[v].append((u, v, w))
            trigs_down[w].append((u, v, w))
        Trigs.append(trigs_down)
    return Adj, Trigs
