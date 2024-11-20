def load_knowledge():
    vecs = []
    vecs.append([0] * 100)
    with open('../../data/kg_embed/entity2vec.vec', 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_ent = torch.FloatTensor(vecs)
    del vecs
    vecs = []
    vecs.append([0] * 100)
    with open('../../data/kg_embed/relation2vec.vec', 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_r = torch.FloatTensor(vecs)
    del vecs
    return embed_ent, embed_r