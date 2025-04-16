@numba.njit
def is_line_segment_intersection_jit(lines1, lines2):
    """check if line segments1 and line segments2 have cross point
    
    Args:
        lines1 (float, [N, 2, 2]): [description]
        lines2 (float, [M, 2, 2]): [description]
    
    Returns:
        [type]: [description]
    """
    N = lines1.shape[0]
    M = lines2.shape[0]
    ret = np.zeros((N, M), dtype=np.bool_)
    for i in range(N):
        for j in range(M):
            A = lines1[i, 0]
            B = lines1[i, 1]
            C = lines2[j, 0]
            D = lines2[j, 1]
            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
            if acd != bcd:
                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] -
                    A[0])
                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] -
                    A[0])
                if abc != abd:
                    ret[i, j] = True
    return ret
