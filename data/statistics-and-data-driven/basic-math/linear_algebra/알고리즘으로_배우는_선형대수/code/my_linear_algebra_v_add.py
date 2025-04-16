def v_add(u, v):
    """
    벡터의 덧셈
    입력값: 더하고자 하는 벡터 u, v
    출력값: 벡터 u, v의 덧셈 결과 w
    """
    n = len(u)
    w = []
    for i in range(0, n):
        val = u[i] + v[i]
        w.append(val)
    return w
