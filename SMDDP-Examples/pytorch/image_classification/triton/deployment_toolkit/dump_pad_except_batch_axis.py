def pad_except_batch_axis(data: np.ndarray, target_shape_with_batch_axis:
    Iterable[int]):
    assert all([(current_size <= target_size) for target_size, current_size in
        zip(target_shape_with_batch_axis, data.shape)]
        ), 'target_shape should have equal or greater all dimensions comparing to data.shape'
    padding = [(0, 0)] + [(0, target_size - current_size) for target_size,
        current_size in zip(target_shape_with_batch_axis[1:], data.shape[1:])]
    return np.pad(data, padding, 'constant', constant_values=np.nan)
