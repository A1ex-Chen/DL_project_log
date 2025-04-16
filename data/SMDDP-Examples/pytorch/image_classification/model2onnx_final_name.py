def final_name(base_name):
    splitted = base_name.split('.')
    if 'pt' in splitted:
        fin_name = base_name.replace('pt', 'onnx')
    elif 'pth' in splitted:
        fin_name = base_name.replace('pth', 'onnx')
    elif len(splitted) > 1:
        fin_name = '.'.join(splitted[:-1] + ['onnx'])
    else:
        fin_name = base_name + '.onnx'
    return fin_name
