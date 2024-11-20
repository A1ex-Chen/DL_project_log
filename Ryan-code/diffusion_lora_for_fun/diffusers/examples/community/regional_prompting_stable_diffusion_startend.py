def startend(cells, array):
    current_start = 0
    array = [float(x) for x in array]
    for value in array:
        end = current_start + value / sum(array)
        cells.append([current_start, end])
        current_start = end
