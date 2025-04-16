@try_export
def export_edgetpu(self, tflite_model='', prefix=colorstr('Edge TPU:')):
    """YOLOv8 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/."""
    LOGGER.warning(
        f'{prefix} WARNING ⚠️ Edge TPU known bug https://github.com/ultralytics/ultralytics/issues/1185'
        )
    cmd = 'edgetpu_compiler --version'
    help_url = 'https://coral.ai/docs/edgetpu/compiler/'
    assert LINUX, f'export only supported on Linux. See {help_url}'
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.
        DEVNULL, shell=True).returncode != 0:
        LOGGER.info(
            f"""
{prefix} export requires Edge TPU compiler. Attempting install from {help_url}"""
            )
        sudo = subprocess.run('sudo --version >/dev/null', shell=True
            ).returncode == 0
        for c in (
            'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -'
            ,
            'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list'
            , 'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
            subprocess.run(c if sudo else c.replace('sudo ', ''), shell=
                True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True
        ).stdout.decode().split()[-1]
    LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
    f = str(tflite_model).replace('.tflite', '_edgetpu.tflite')
    cmd = (
        f'edgetpu_compiler -s -d -k 10 --out_dir "{Path(f).parent}" "{tflite_model}"'
        )
    LOGGER.info(f"{prefix} running '{cmd}'")
    subprocess.run(cmd, shell=True)
    self._add_tflite_metadata(f)
    return f, None
