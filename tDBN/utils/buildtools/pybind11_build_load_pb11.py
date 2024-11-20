def load_pb11(sources, target, cwd='.', cuda=False, arch=None, num_workers=
    4, includes: list=None, build_directory=None, compiler='g++'):
    cmd_groups = []
    cmds = []
    outs = []
    main_sources = []
    if arch is None:
        arch = find_cuda_device_arch()
    for s in sources:
        s = str(s)
        if '.cu' in s or '.cu.cc' in s:
            assert cuda is True, 'cuda must be true if contain cuda file'
            cmds.append(Nvcc(s, out(s), arch))
            outs.append(out(s))
        else:
            main_sources.append(s)
    if cuda is True and arch is None:
        raise ValueError('you must specify arch if sources contains cuda files'
            )
    cmd_groups.append(cmds)
    if cuda:
        cmd_groups.append([Pybind11CUDALink(outs + main_sources, target,
            includes=includes)])
    else:
        cmd_groups.append([Pybind11Link(outs + main_sources, target,
            includes=includes)])
    for cmds in cmd_groups:
        compile_libraries(cmds, cwd, num_workers=num_workers, compiler=compiler
            )
    return import_file(target, add_to_sys=False, disable_warning=True)
