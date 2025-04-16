def matmul(A, B):
    """
    행렬의 행렬곱
    입력값: 행렬곱을 수행할 행렬 A, B
    출력값: 행렬 A와 행렬 B의 행렬곱 결과인 행렬 res
    """
    n = len(A)
    p1 = len(A[0])
    p2 = len(B[0])
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p2):
            val = 0
            for k in range(0, p1):
                val += A[i][k] * B[k][j]
            row.append(val)
        res.append(row)
    return res
