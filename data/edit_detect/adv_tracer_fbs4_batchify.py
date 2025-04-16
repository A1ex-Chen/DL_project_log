def batchify(data, batch_size: int):
    res: list = []
    for i in range(0, len(data), batch_size):
        res.append(data[i:i + batch_size])
    return res
