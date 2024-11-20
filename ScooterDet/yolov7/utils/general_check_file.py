def check_file(file):
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(files), f'File Not Found: {file}'
        assert len(files
            ) == 1, f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]
