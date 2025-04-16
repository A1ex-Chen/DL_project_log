def u_bidiag(A):
    """
    upper bidiagonal 행렬
    입력값: 행렬 A
    출력값: 행렬 A의 upper bidiagonal 행렬 res
    """
    n = len(A)
    p = len(A[0])
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i > j or j - i > 1:
                row.append(0)
            else:
                row.append(A[i][j])
        res.append(row)
    return res
