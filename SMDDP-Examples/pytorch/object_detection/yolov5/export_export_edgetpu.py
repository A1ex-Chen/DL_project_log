def export_edgetpu(file, prefix=colorstr('Edge TPU:')):
    try:
        cmd = 'edgetpu_compiler --version'
        help_url = 'https://coral.ai/docs/edgetpu/compiler/'
        assert platform.system(
            ) == 'Linux', f'export only supported on Linux. See {help_url}'
        if subprocess.run(f'{cmd} >/dev/null', shell=True).returncode != 0:
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
                , 'sudo apt-get update',
                'sudo apt-get install edgetpu-compiler'):
                subprocess.run(c if sudo else c.replace('sudo ', ''), shell
                    =True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True
            ).stdout.decode().split()[-1]
        LOGGER.info(
            f'\n{prefix} starting export with Edge TPU compiler {ver}...')
        f = str(file).replace('.pt', '-int8_edgetpu.tflite')
        f_tfl = str(file).replace('.pt', '-int8.tflite')
        cmd = f'edgetpu_compiler -s -d -k 10 --out_dir {file.parent} {f_tfl}'
        subprocess.run(cmd.split(), check=True)
        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
