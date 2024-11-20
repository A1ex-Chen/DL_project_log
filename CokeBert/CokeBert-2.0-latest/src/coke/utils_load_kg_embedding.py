def load_kg_embedding(data_dir):
    """Load KG embedding"""
    vecs = []
    vecs.append([0] * 100)
    with open(os.path.join(data_dir, 'kg_embed/entity2vec.vec'), 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_ent = torch.FloatTensor(vecs)
    del vecs
    vecs = []
    vecs.append([0] * 100)
    with open(os.path.join(data_dir, 'kg_embed/relation2vec.vec'), 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_r = torch.FloatTensor(vecs)
    del vecs
    return embed_ent, embed_r
