def _np_array_split(a, v):
    """Split numpy array by separator value.

  Args:
    a: 1-D numpy.array.
    v: number. Separator value. e.g -1.

  Returns:
    2-D list of clean separated arrays.

  Example:
    a = [1, 2, 3, 4, -1, 5, 6, 7, 8]
    b = _np_array_split(a, -1)
    # Output: b = [[1, 2, 3, 4], [5, 6, 7, 8]]
  """
    a = np.array(a)
    arrs = np.split(a, np.where(a[:] == v)[0])
    return [(e if len(e) <= 0 or e[0] != v else e[1:]) for e in arrs]
