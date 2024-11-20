def ele2diag(a):
    """
    대각원소 -> 대각행렬 변환
    입력값: 대각 원소 리스트 a
    출력값: 대각 원소 a를 이용해 생성한 nxn 대각 행렬 D
    n: 대각 원소 리스트 a의 길이
    """
    n = len(a)
    D = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i == j:
                row.append(a[i])
            else:
                row.append(0)
        D.append(row)
    return D
