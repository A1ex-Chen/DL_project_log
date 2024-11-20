def sample(ids_list):
    return np.random.choice(ids_list, len(ids_list) * 3 // 5, replace=False)
