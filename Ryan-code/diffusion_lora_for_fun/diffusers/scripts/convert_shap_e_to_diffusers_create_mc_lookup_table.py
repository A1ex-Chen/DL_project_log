def create_mc_lookup_table():
    cases = torch.zeros(256, 5, 3, dtype=torch.long)
    masks = torch.zeros(256, 5, dtype=torch.bool)
    edge_to_index = {(0, 1): 0, (2, 3): 1, (4, 5): 2, (6, 7): 3, (0, 2): 4,
        (1, 3): 5, (4, 6): 6, (5, 7): 7, (0, 4): 8, (1, 5): 9, (2, 6): 10,
        (3, 7): 11}
    for i, case in enumerate(MC_TABLE):
        for j, tri in enumerate(case):
            for k, (c1, c2) in enumerate(zip(tri[::2], tri[1::2])):
                cases[i, j, k] = edge_to_index[(c1, c2) if c1 < c2 else (c2,
                    c1)]
            masks[i, j] = True
    return cases, masks
