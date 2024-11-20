def openFile(f):
    try:
        d = open(f, 'r')
        return d
    except IOError:
        print('Error opening file {}. Exiting.'.format(f), file=sys.stderr)
        sys.exit(1)
