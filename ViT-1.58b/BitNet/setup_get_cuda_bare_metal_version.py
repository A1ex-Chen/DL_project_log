def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + '/bin/nvcc', '-V'],
        universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index('release') + 1
    bare_metal_version = parse(output[release_idx].split(',')[0])
    return raw_output, bare_metal_version
