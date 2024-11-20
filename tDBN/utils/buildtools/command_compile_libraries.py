def compile_libraries(cmds, code_folder=None, compiler: str=None,
    num_workers=-1):
    if num_workers == -1:
        num_workers = min(len(cmds), multiprocessing.cpu_count())
    if num_workers == 0:
        rets = map(partial(compile_func, code_folder=code_folder, compiler=
            compiler), cmds)
    else:
        with ProcessPoolExecutor(num_workers) as pool:
            func = partial(compile_func, code_folder=code_folder, compiler=
                compiler)
            rets = pool.map(func, cmds)
    if any([(r.returncode != 0) for r in rets]):
        cmds.clear()
        return False
    cmds.clear()
    return True
