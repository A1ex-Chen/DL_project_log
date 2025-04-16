def nested_expand_like(arrays, new_seq_length, padding_index=-100):
    """ Expand the `arrays` so that the second dimension grows to `new_seq_length`. Uses `padding_index` for padding."""
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(nested_expand_like(x, new_seq_length,
            padding_index=padding_index) for x in arrays)
    result = np.full_like(arrays, padding_index, shape=(arrays.shape[0],
        new_seq_length) + arrays.shape[2:])
    result[:, :arrays.shape[1]] = arrays
    return result
