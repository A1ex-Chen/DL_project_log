def int_to_alphabetic(integer):
    s = ''
    while integer != 0:
        s = chr(ord('A') + integer % 10) + s
        integer //= 10
    return s
