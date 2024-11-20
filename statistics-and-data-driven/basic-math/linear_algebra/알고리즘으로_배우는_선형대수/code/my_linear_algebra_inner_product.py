def inner_product(a, b):
    """
    벡터의 내적
    입력값: 내적할 벡터 리스트 a, b
    출력값: 벡터 a, b의 내적 결과 res
    """
    n = len(a)
    res = 0
    for i in range(0, n):
        res += a[i] * b[i]
    return res
