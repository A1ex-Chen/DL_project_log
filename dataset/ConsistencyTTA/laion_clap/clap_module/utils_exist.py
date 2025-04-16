def exist(dataset_name, dataset_type):
    """
    Check if dataset exists
    """
    if dataset_type in dataset_split[dataset_name]:
        return True
    else:
        return False
