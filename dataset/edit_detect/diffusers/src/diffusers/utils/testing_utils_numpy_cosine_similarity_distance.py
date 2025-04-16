def numpy_cosine_similarity_distance(a, b):
    similarity = np.dot(a, b) / (norm(a) * norm(b))
    distance = 1.0 - similarity.mean()
    return distance
