def inter_distances(tensors: torch.Tensor):
    """
            To calculate the distance between each two depth maps.
            """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i:i + 1]
        arr2 = tensors[j:j + 1]
        distances.append(arr1 - arr2)
    dist = torch.concatenate(distances, dim=0)
    return dist
