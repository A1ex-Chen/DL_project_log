def quant_sensitivity_load(file):
    assert os.path.exists(file), print('File {} does not exist'.format(file))
    quant_sensitivity = list()
    with open(file, 'r') as qfile:
        lines = qfile.readlines()
        for line in lines:
            layer, mAP1, mAP2 = line.strip('\n').split(' ')
            quant_sensitivity.append((layer, float(mAP1), float(mAP2)))
    return quant_sensitivity
