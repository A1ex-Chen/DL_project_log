def qr_gram(A):
    """
    그램 슈미트 방법을 이용한 QR분해
    입력값: 행렬 A
    출력값: 행렬 A를 그램 슈미트 방법을 이용해 QR분해한 결과 행렬 Q, R
    """
    n = len(A)
    p = len(A[0])
    At = transpose(A)
    U = []
    norm_list = []
    V = []
    Q = []
    R = []
    for i in range(0, n):
        if i == 0:
            u = At[i]
            norm_u = norm(u)
            U.append(u)
            norm_list.append(norm_u)
        else:
            a = At[i]
            dp_list = []
            for j in range(0, i):
                dp = inner_product(a, U[j])
                dp_list.append(dp)
            u = []
            for j in range(0, n):
                val = a[j]
                for k in range(0, i):
                    val -= dp_list[k] / norm_list[k] ** 2 * U[k][j]
                u.append(val)
            norm_u = norm(u)
            U.append(u)
            norm_list.append(norm_u)
        v = normalize(u)
        V.append(v)
    Q = transpose(V)
    for i in range(0, n):
        r = []
        for j in range(0, n):
            if i > j:
                r.append(0)
            else:
                r_ele = inner_product(At[j], V[i])
                r.append(r_ele)
        R.append(r)
    return Q, R
