def repeat_div(x, y):
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x
