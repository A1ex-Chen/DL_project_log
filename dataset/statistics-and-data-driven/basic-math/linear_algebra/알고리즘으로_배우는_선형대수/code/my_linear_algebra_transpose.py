def transpose(A):
    """
    행렬의 전치행렬
    입력값: 전치행렬을 구하고자 하는 행렬 A
    출력값: 행렬 A의 전치행렬 At
    """
    n = len(A)
    p = len(A[0])
    At = []
    for i in range(0, p):
        row = []
        for j in range(0, n):
            val = A[j][i]
            row.append(val)
        At.append(row)
    return At
