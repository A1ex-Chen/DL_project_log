def identity(n):
    """
    항등행렬 생성
    입력값: 항등 행렬의 크기 n
    출력값: nxn 항등 행렬 I
    """
    I = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        I.append(row)
    return I
