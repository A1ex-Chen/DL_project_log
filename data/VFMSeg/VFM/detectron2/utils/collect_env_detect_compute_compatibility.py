def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, 'bin', 'cuobjdump')
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output("'{}' --list-elf '{}'".format(
                cuobjdump, so_file), shell=True)
            output = output.decode('utf-8').strip().split('\n')
            arch = []
            for line in output:
                line = re.findall('\\.sm_([0-9]*)\\.', line)[0]
                arch.append('.'.join(line))
            arch = sorted(set(arch))
            return ', '.join(arch)
        else:
            return so_file + '; cannot find cuobjdump'
    except Exception:
        return so_file
