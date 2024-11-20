def gsutil_getsize(url=''):
    output = subprocess.check_output(['gsutil', 'du', url], shell=True,
        encoding='utf-8')
    if output:
        return int(output.split()[0])
    return 0
