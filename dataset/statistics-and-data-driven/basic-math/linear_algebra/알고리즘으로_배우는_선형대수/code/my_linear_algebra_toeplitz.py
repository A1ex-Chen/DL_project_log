def toeplitz(a, b):
    """
    토플리츠 행렬 변환
    입력값: 토플리츠 행렬로 변환하고자 하는 리스트 a, b
    출력값: 리스트 a, b를 이용해 만든 토플리츠 행렬 A
    """
    n1 = len(a)
    n2 = len(b)
    A = []
    for i in range(0, n1):
        row = []
        for j in range(0, n2):
            if i > j:
                row.append(a[i - j])
            else:
                row.append(b[j - i])
        A.append(row)
    return A
