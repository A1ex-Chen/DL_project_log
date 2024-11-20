def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + '/bin/nvcc', '-V'],
        universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index('release') + 1
    release = output[release_idx].split('.')
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    torch_binary_major = torch.version.cuda.split('.')[0]
    torch_binary_minor = torch.version.cuda.split('.')[1]
    print('\nCompiling cuda extensions with')
    print(raw_output + 'from ' + cuda_dir + '/bin\n')
    if (bare_metal_major != torch_binary_major or bare_metal_minor !=
        torch_binary_minor):
        raise RuntimeError(
            'Cuda extensions are being compiled with a version of Cuda that does '
             + 'not match the version used to compile Pytorch binaries.  ' +
            """Pytorch binaries were compiled with Cuda {}.
""".format(
            torch.version.cuda) +
            'In some cases, a minor-version mismatch will not cause later errors:  '
             +
            'https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  You can try commenting out this check (at your own risk).'
            )
