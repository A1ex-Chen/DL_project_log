def compile_func(cmd, code_folder, compiler):
    if not isinstance(cmd, (Link, Nvcc)):
        shell = cmd.shell(compiler=compiler)
    else:
        shell = cmd.shell()
    print(shell)
    cwd = None
    if code_folder is not None:
        cwd = str(code_folder)
    ret = subprocess.run(shell, shell=True, cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError('compile failed with retcode', ret.returncode)
    return ret
