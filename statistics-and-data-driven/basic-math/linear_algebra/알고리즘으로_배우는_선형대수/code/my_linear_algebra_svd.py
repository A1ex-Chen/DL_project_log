def svd(A):
    """
    svd를 이용한 고윳값, 고유벡터 구하기
    입력값: 고윳값, 고유벡터를 구하고자 하는 행렬 A
    출력값: U = 고유벡터, S = 특이값, Vt = AtA의 고유벡터
    """
    At = transpose(A)
    AtA = matmul(At, A)
    E, V = eig_qr(AtA)
    n = len(AtA)
    for i in range(0, n):
        E[i][i] = E[i][i] ** 0.5
    S = diag(E)
    Vt = transpose(V)
    AV = matmul(A, V)
    AVt = transpose(AV)
    Ut = []
    for vector in AVt:
        Ut.append(normalize(vector))
    U = transpose(Ut)
    return U, S, Vt
