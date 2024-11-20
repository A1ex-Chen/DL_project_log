def getData():
    import re
    f = open('C:/Users/xuwei/Desktop/data.txt', 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    random.shuffle(data)
    train_data = data[:int(len(data) * 7 / 10)]
    test_data = data[int(len(data) * 7 / 10):]
    print('load data finished')
    print('total data ', len(data))
    return train_data, test_data
