def run_ruff(code):
    command = ['ruff', 'format', '-', '--config', 'pyproject.toml', '--silent']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=
        subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, _ = process.communicate(input=code.encode())
    return stdout.decode()