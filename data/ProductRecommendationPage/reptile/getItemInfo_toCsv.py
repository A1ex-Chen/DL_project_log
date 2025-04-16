def toCsv(ilt, name):
    data = pd.DataFrame(ilt, columns=['序号', '价格', '名称', '图片URL', '类型', '销量'])
    data.to_csv(f'../data/{name}.csv', index=False)
    print(data.head())
    return data
