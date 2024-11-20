def make_cells(ratios):
    if ';' not in ratios and ',' in ratios:
        ratios = ratios.replace(',', ';')
    ratios = ratios.split(';')
    ratios = [inratios.split(',') for inratios in ratios]
    icells = []
    ocells = []

    def startend(cells, array):
        current_start = 0
        array = [float(x) for x in array]
        for value in array:
            end = current_start + value / sum(array)
            cells.append([current_start, end])
            current_start = end
    startend(ocells, [r[0] for r in ratios])
    for inratios in ratios:
        if 2 > len(inratios):
            icells.append([[0, 1]])
        else:
            add = []
            startend(add, inratios[1:])
            icells.append(add)
    return ocells, icells, sum(len(cell) for cell in icells)
