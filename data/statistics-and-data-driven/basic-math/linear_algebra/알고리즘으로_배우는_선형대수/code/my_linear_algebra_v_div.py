def v_div(u, v):
    """
    벡터의 원소 나눗셈
    입력값: 원소 나눗셈 하고자할 벡터 리스트 u, v
    출력값: 벡터 u, v의 원소 나눗셈 결과 w
    """
    n = len(u)
    w = []
    for i in range(0, n):
        val = u[i] / v[i]
        w.append(val)
    return w
