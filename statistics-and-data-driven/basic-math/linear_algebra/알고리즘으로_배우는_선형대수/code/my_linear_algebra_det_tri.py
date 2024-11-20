def det_tri(A):
    """
    상 삼각 행렬 변환을 이용해 행렬식 구하기
    입력값: 행렬 A
    출력값: 행렬식 res
    """
    n = len(A)
    X = deepcopy(A)
    n_row_change = 0
    for i in range(0, n):
        if X[i][i] == 0:
            tmp = X[i + 1]
            X[i + 1] = X[i]
            X[i] = tmp
            n_row_change += 1
        for j in range(i + 1, n):
            ratio = X[j][i] / X[i][i]
            for k in range(0, n):
                X[j][k] = X[j][k] - ratio * X[i][k]
    n_row_change = (-1) ** n_row_change
    res = 1
    for i in range(n):
        res *= X[i][i]
    res *= n_row_change
    return res
