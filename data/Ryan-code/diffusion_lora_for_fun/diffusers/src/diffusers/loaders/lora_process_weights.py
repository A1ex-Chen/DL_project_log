def process_weights(adapter_names, weights):
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)
    if len(adapter_names) != len(weights):
        raise ValueError(
            f'Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}'
            )
    weights = [(w if w is not None else 1.0) for w in weights]
    return weights
