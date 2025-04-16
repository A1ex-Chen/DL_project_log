def quant_sensitivity_save(quant_sensitivity, file):
    with open(file, 'w') as qfile:
        for item in quant_sensitivity:
            name, mAP1, mAP2 = item
            line = name + ' ' + '{:0.4f}'.format(mAP1
                ) + ' ' + '{:0.4f}'.format(mAP2) + '\n'
            qfile.write(line)
