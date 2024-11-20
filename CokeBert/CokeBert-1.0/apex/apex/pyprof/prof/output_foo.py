def foo(self, cadena, pformat):
    if self.csv:
        cadena = ','.join(map(lambda x: '"' + str(x) + '"', cadena))
    elif self.col:
        cadena = pformat % cadena
    else:
        cadena = ' '.join(map(str, cadena))
    try:
        print(cadena)
    except IOError as e:
        if e.errno == errno.EPIPE:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(0)
        else:
            sys.exit(-1)
