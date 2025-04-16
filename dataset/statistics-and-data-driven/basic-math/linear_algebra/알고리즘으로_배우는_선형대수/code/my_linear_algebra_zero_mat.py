def zero_mat(n, p):
    """
    영 행렬 생성
    입력값: 생성하고자 할 영 행렬의 크기 n행, p열
    출력값: nxp 영 행렬 Z
    """
    Z = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            row.append(0)
        Z.append(row)
    return Z
