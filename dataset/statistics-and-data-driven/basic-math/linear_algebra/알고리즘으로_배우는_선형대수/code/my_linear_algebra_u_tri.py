def u_tri(A):
    """
    상 삼각 행렬 변환
    입력값: 상 삼각 행렬로 변환하고자 하는 행렬 A
    출력값: 행렬 A를 상 삼각 행렬로 변환시킨 행렬 utri
    """
    n = len(A)
    p = len(A[0])
    utri = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i > j:
                row.append(0)
            else:
                row.append(A[i][j])
        utri.append(row)
    return utri
