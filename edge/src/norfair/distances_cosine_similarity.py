def cosine_similarity(emb1, emb2):
    return (1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm
        (emb2))) / 2
