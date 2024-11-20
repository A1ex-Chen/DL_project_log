def device_count():
    assert platform.system() in ('Linux', 'Windows'
        ), 'device_count() only supported on Linux or Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system(
            ) == 'Linux' else 'nvidia-smi -L | find /c /v ""'
        return int(subprocess.run(cmd, shell=True, capture_output=True,
            check=True).stdout.decode().split()[-1])
    except Exception:
        return 0
