def sign(a):
    """
    스칼라 a의 부호
    입력값: 스칼라 a
    출력값: 스칼라 a가 a>=0면 1 출력, a<0이면 0 출력
    """
    res = 1
    if a < 0:
        res = -1
    return res
