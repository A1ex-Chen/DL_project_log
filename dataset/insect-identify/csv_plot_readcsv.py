def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x, y
