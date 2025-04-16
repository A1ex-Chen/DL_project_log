def inv(A):
    """
    행렬 A의 역행렬 구하기
    입력값: 행렬 A
    출력값: 행렬 A의 역행렬 res
    """
    n = len(A)
    X = deepcopy(A)
    C = []
    for i in range(0, n):
        row_C = []
        idx_r = list(range(0, n))
        idx_r.remove(i)
        for j in range(0, n):
            idx_c = list(range(0, n))
            idx_c.remove(j)
            M = []
            for k in idx_r:
                row_M = []
                for l in idx_c:
                    val = X[k][l]
                    row_M.append(val)
                M.append(row_M)
            Mij = det_rec(M)
            Cij = (-1) ** (i + j) * Mij
            row_C.append(Cij)
        C.append(row_C)
    adj = transpose(C)
    res = scalar_mul(1 / det_rec(X), adj)
    return res
