def eig_qr(A):
    """
    이 방법은 행렬 A가 대칭행렬이여야만 사용할수있음
    QR분해를 이용한 고윳값, 고유벡터 구하기
    인풋: 고윳값, 고유벡터를 구하고자 하는 행렬 A
    아웃풋: E = 고윳값, V = 고유벡터
    """
    n = len(A)
    E = deepcopy(A)
    V = identity(n)
    for i in range(0, 30):
        Q, R = qr_gram(E)
        E = matmul(R, Q)
        V = matmul(V, Q)
    return E, V
