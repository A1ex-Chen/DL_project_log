def outer_product(a, b):
    """
    벡터의 외적
    입력값: 외적할 벡터 리스트 a, b
    출력값: 벡터 a, b의 외적 결과 res
    """
    res = []
    n1 = len(a)
    n2 = len(b)
    for i in range(0, n1):
        row = []
        for j in range(0, n2):
            val = a[i] * b[j]
            row.append(val)
        res.append(row)
    return res
