def ele_product(A, B):
    """
    행렬의 원소곱
    입력값: 행렬의 원소곱을 수행할 행렬 A, B
    출력값: 행렬 A와 행렬 B의 원소곱 결과인 행렬 res
    """
    n = len(A)
    p = len(A[0])
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] * B[i][j]
            row.append(val)
        res.append(row)
    return res
