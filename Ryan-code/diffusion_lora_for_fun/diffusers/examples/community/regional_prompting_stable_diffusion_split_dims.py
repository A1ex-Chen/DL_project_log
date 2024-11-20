def split_dims(xs, height, width):
    xs = xs

    def repeat_div(x, y):
        while y > 0:
            x = math.ceil(x / 2)
            y = y - 1
        return x
    scale = math.ceil(math.log2(math.sqrt(height * width / xs)))
    dsh = repeat_div(height, scale)
    dsw = repeat_div(width, scale)
    return dsh, dsw
