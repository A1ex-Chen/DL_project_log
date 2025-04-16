def scalar_mul(b, A):
    """
    행렬의 스칼라곱
    입력값: 스칼라곱을 수행할 스칼라 b, 행렬 A
    출력값: 스칼라 b와 행렬 A의 스칼라 곱 결과인 행렬 res
    """
    n = len(A)
    p = len(A[0])
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = b * A[i][j]
            row.append(val)
        res.append(row)
    return res
