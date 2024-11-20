def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), dim=0)
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1],
        array2.shape[1])) + array1.shape[2:]
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[:array1.shape[0], :array1.shape[1]] = array1
    result[array1.shape[0]:, :array2.shape[1]] = array2
    return result
