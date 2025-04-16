def qr_householder(A):
    """
    행렬 A의 하우스홀더 방법을 이용한 QR분해
    입력값: 행렬 A
    출력값: 행렬 Q, 행렬 R
    """
    n = len(A)
    p = len(A[0])
    H_list = []
    for i in range(0, p):
        if i == 0:
            A1 = deepcopy(A)
            exA = A1
        elif i < p - 1:
            Ai = []
            for j in range(1, len(exA)):
                row = []
                for k in range(1, len(exA[0])):
                    row.append(HA[j][k])
                Ai.append(row)
            exA = Ai
        elif i == p - 1:
            Ap = []
            for j in range(1, len(HA)):
                Ap.append(HA[j][1])
            exA = Ap
        if i < p - 1:
            a = transpose(exA)[0]
        else:
            a = exA
        nm = norm(a)
        e = [1]
        for j in range(0, len(a) - 1):
            e.append(0)
        tmp_e = []
        for j in range(0, len(e)):
            val = sign(a[0]) * nm * e[j]
            tmp_e.append(val)
        v = v_add(a, tmp_e)
        H = householder(v)
        if i == p - 1:
            HA = []
            for j in range(0, len(H)):
                val = 0
                for k in range(0, len(H[0])):
                    val += H[j][k] * exA[k]
                HA.append(val)
        else:
            HA = matmul(H, exA)
        H_list.append(H)
        if i > 0:
            tmp_H = identity(len(A))
            for j in range(i, len(A)):
                for k in range(i, len(A)):
                    tmp_H[j][k] = H_list[-1][j - i][k - i]
            H_list[-1] = tmp_H
    Q = deepcopy(H_list[0])
    for j in range(0, len(H_list) - 1):
        Q = matmul(Q, H_list[j + 1])
    R = deepcopy(H_list[-1])
    for j in range(1, len(H_list)):
        R = matmul(R, H_list[-(j + 1)])
    R = matmul(R, A)
    return Q, R
