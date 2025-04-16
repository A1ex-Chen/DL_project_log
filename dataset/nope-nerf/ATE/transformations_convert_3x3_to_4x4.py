def convert_3x3_to_4x4(matrix_3x3):
    M = numpy.identity(4)
    M[:3, :3] = matrix_3x3
    return M
