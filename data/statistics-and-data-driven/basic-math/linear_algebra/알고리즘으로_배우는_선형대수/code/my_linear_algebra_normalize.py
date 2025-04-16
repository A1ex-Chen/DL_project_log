def normalize(a):
    """
    벡터 a의 normalization
    벡터 a의 norm을 1로 만들어줌
    입력값: normalization할 벡터 리스트 a
    출력값: 벡터 a를 normalization한 결과 벡터 리스트 v
    """
    n = len(a)
    v = []
    for i in range(0, n):
        tmp = a[i] / norm(a)
        v.append(tmp)
    return v
