def det_rec(A):
    """
    행렬 A의 행렬식 구하기(재귀 방식을 이용)
    입력값: 행렬식을 구하고자 하는 행렬 A
    출력값: 행렬 A의 행렬식 res
    """
    n = len(A)
    res = 0
    if n == 2:
        res = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return res
    for i in range(0, n):
        X = deepcopy(A)
        X = X[1:]
        nx = len(X)
        for j in range(0, nx):
            X[j] = X[j][0:i] + X[j][i + 1:]
        sign = (-1) ** (i % 2)
        sub_det = det_rec(X)
        res += sign * A[0][i] * sub_det
    return res
