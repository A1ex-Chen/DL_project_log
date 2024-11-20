def write_txt_file(ordered_tgt, path):
    f = Path(path).open('w')
    for ln in ordered_tgt:
        f.write(ln + '\n')
        f.flush()
