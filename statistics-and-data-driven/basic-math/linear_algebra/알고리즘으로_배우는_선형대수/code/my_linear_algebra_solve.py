def solve(A, b):
    """
    연립 방정식 풀기
    입력값: 솔루션을 구하고 싶은 A, b
    출력값: 방정식의 솔루션 sol
    """
    X = deepcopy(A)
    sol = deepcopy(b)
    n = len(X)
    for i in range(0, n):
        x_row = X[i]
        y_val = sol[i]
        if x_row[i] != 0:
            tmp = 1 / x_row[i]
        else:
            tmp = 0
        x_row = [(element * tmp) for element in x_row]
        y_val = y_val * tmp
        X[i] = x_row
        sol[i] = y_val
        for j in range(0, n):
            if i == j:
                continue
            x_next = X[j]
            y_next = sol[j]
            x_tmp = [(element * -x_next[i]) for element in x_row]
            y_tmp = y_val * -x_next[i]
            for k in range(0, len(x_row)):
                x_next[k] = x_tmp[k] + x_next[k]
            y_next = y_tmp + y_next
            X[j] = x_next
            sol[j] = y_next
    return sol
