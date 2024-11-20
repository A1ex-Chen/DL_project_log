@Retry(times=2, delay=1)
def attempt_install(packages, commands):
    """Attempt pip install command with retries on failure."""
    return subprocess.check_output(
        f'pip install --no-cache-dir {packages} {commands}', shell=True
        ).decode()
