def tee(line, sink, pipe, label=''):
    line = line.decode('utf-8').rstrip()
    sink.append(line)
    if not quiet:
        print(label, line, file=pipe)
