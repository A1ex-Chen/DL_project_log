def diag(A):
    """
    행렬의 대각행렬
    입력값: 대각행렬을 구하고자 하는 행렬 A
    출력값: 행렬 A의 대각행렬 D
    """
    n = len(A)
    D = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i == j:
                row.append(A[i][j])
            else:
                row.append(0)
        D.append(row)
    return D
