def scalar_v_mul(a, u):
    """
    벡터의 스칼라 곱
    입력값: 스칼라 a, 벡터 리스트 u
    출력값: 스칼라 a와 벡터 리스트 u의 곱 결과 w
    """
    n = len(u)
    w = []
    for i in range(0, n):
        val = a * u[i]
        w.append(val)
    return w
