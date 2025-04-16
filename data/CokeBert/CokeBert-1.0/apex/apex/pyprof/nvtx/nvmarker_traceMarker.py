def traceMarker(stack):
    d = {}
    cadena = []
    for i in range(len(stack) - 1):
        fi = stack[i]
        t = '{}:{}'.format(fi.filename, fi.lineno)
        cadena.append(t)
    d['traceMarker'] = cadena
    return str(d)
