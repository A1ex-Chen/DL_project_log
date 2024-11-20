def _parse(weights):
    gene = []
    n = 2
    start = 0
    for i in range(self.num_nodes):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in
            range(len(W[x])) if k != self.primitives.index('none')))[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k != self.primitives.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((self.primitives[k_best], j))
        start = end
        n += 1
    return gene
