def sample(ids_list):
    return np.random.choice(ids_list, len(ids_list) // 2, replace=False)
