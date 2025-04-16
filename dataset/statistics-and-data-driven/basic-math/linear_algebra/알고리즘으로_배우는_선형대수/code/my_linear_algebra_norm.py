def norm(a):
    """
    벡터의 norm
    입력값: norm을 구하고자 할 벡터 a
    출력값: 벡터 a의 norm 결과 res
    """
    n = len(a)
    res = 0
    for i in range(0, n):
        res += a[i] ** 2
    res = res ** 0.5
    return res
