def replace_in_files(path_to_datafile):
    with open(path_to_datafile) as datafile:
        lines_to_copy = []
        skip_file = False
        skip_snippet = False
        for line in datafile:
            if '# To replace in: ' in line and '##' not in line:
                file_to_replace_in = line.split('"')[1]
                skip_file = skip_units(line)
            elif '# Below: ' in line and '##' not in line:
                line_to_copy_below = line.split('"')[1]
                skip_snippet = skip_units(line)
            elif '# End.' in line and '##' not in line:
                if not skip_file and not skip_snippet:
                    replace(file_to_replace_in, line_to_copy_below,
                        lines_to_copy)
                lines_to_copy = []
            elif '# Replace with' in line and '##' not in line:
                lines_to_copy = []
            elif '##' not in line:
                lines_to_copy.append(line)
    remove(path_to_datafile)
