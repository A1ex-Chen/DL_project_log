def genotype(self):
    """
        :return:
        """

    def _parse(weights):
        """
            :param weights: [14, 8]
            :return:
            """
        gene = []
        n = 2
        start = 0
        for i in range(self.num_nodes):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in
                range(len(W[x])) if k != LINEAR_PRIMITIVES.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != LINEAR_PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((LINEAR_PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene
    gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().
        numpy())
    gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().
        numpy())
    concat = range(2 + self.num_nodes - self.channel_multiplier, self.
        num_nodes + 2)
    genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=
        gene_reduce, reduce_concat=concat)
    return genotype
