def lu_decomp(A):
    """
    LU 분해
    입력값: 행렬 A
    출력값: 행렬 A의 LU분해 행렬 L, U
    """
    n = len(A)
    p = len(A[0])
    L = [([0] * p) for i in range(0, n)]
    U = []
    for i in range(0, n):
        a = A[i]
        val = 1 / a[i]
        L[i][i] = 1 / val
        a = [(element * val) for element in a]
        U.append(a)
        for j in range(i + 1, n):
            row = A[j]
            a_tmp = [(element * -row[i]) for element in a]
            L[j][i] = row[i]
            A[j] = [(a_tmp[k] + row[k]) for k in range(p)]
    return L, U
