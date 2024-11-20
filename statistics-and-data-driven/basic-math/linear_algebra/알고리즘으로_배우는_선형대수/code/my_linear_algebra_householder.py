def householder(v):
    """
    하우스홀더 행렬
    입력값: 하우스홀더 행렬을 생성할 리스트 v
    출력값: 리스트 v를 이용해 생성한 하우스홀더 행렬 H
    """
    n = len(v)
    outer_mat = outer_product(v, v)
    inner_val = inner_product(v, v)
    V = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            val = 2 / inner_val * outer_mat[i][j]
            row.append(val)
        V.append(row)
    H = subtract(identity(n), V)
    return H
