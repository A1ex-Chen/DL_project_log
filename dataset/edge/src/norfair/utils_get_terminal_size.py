def get_terminal_size(default: Tuple[int, int]=(80, 24)) ->Tuple[int, int]:
    columns, lines = default
    for fd in range(0, 3):
        try:
            columns, lines = os.get_terminal_size(fd)
        except OSError:
            continue
        break
    return columns, lines
