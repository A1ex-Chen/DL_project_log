def diag_ele(A):
    """
    대각 원소 구하기
    입력값: 대각 원소를 구하고자 할 행렬 A
    출력값: 행렬 A의 대각 원소 리스트 d
    """
    n = len(A)
    d = []
    for i in range(0, n):
        d.append(A[i][i])
    return d
