def v_subtract(u, v):
    """
    벡터의 뺄셈
    입력값: 빼고자하는 벡터 리스트 u,v
    출력값: 벡터 u,v의 뺄셈 결과 w
    """
    n = len(u)
    w = []
    for i in range(0, n):
        val = u[i] - v[i]
        w.append(val)
    return w
