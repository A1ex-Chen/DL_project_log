def printGoodsList(ilt):
    tply = '{:4}\t{:8}\t{:16}'
    print(tply.format('序号', '价格', '商品价格', '图片URL'))
    count = 0
    for g in ilt:
        count = count + 1
        print(tply.format(count, g[0], g[1], g[2]))
