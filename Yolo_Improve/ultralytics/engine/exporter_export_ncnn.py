@try_export
def export_ncnn(self, prefix=colorstr('NCNN:')):
    """
        YOLOv8 NCNN export using PNNX https://github.com/pnnx/pnnx.
        """
    check_requirements('ncnn')
    import ncnn
    LOGGER.info(f'\n{prefix} starting export with NCNN {ncnn.__version__}...')
    f = Path(str(self.file).replace(self.file.suffix, f'_ncnn_model{os.sep}'))
    f_ts = self.file.with_suffix('.torchscript')
    name = Path('pnnx.exe' if WINDOWS else 'pnnx')
    pnnx = name if name.is_file() else ROOT / name
    if not pnnx.is_file():
        LOGGER.warning(
            f"""{prefix} WARNING ⚠️ PNNX not found. Attempting to download binary file from https://github.com/pnnx/pnnx/.
Note PNNX Binary file must be placed in current working directory or in {ROOT}. See PNNX repo for full installation instructions."""
            )
        system = ('macos' if MACOS else 'windows' if WINDOWS else 
            'linux-aarch64' if ARM64 else 'linux')
        try:
            release, assets = get_github_assets(repo='pnnx/pnnx')
            asset = [x for x in assets if f'{system}.zip' in x][0]
            assert isinstance(asset, str
                ), 'Unable to retrieve PNNX repo assets'
            LOGGER.info(
                f'{prefix} successfully found latest PNNX asset file {asset}')
        except Exception as e:
            release = '20240410'
            asset = f'pnnx-{release}-{system}.zip'
            LOGGER.warning(
                f'{prefix} WARNING ⚠️ PNNX GitHub assets not found: {e}, using default {asset}'
                )
        unzip_dir = safe_download(
            f'https://github.com/pnnx/pnnx/releases/download/{release}/{asset}'
            , delete=True)
        if check_is_path_safe(Path.cwd(), unzip_dir):
            (unzip_dir / name).rename(pnnx)
            pnnx.chmod(511)
            shutil.rmtree(unzip_dir)
    ncnn_args = [f"ncnnparam={f / 'model.ncnn.param'}",
        f"ncnnbin={f / 'model.ncnn.bin'}", f"ncnnpy={f / 'model_ncnn.py'}"]
    pnnx_args = [f"pnnxparam={f / 'model.pnnx.param'}",
        f"pnnxbin={f / 'model.pnnx.bin'}", f"pnnxpy={f / 'model_pnnx.py'}",
        f"pnnxonnx={f / 'model.pnnx.onnx'}"]
    cmd = [str(pnnx), str(f_ts), *ncnn_args, *pnnx_args,
        f'fp16={int(self.args.half)}', f'device={self.device.type}',
        f'inputshape="{[self.args.batch, 3, *self.imgsz]}"']
    f.mkdir(exist_ok=True)
    LOGGER.info(f"{prefix} running '{' '.join(cmd)}'")
    subprocess.run(cmd, check=True)
    pnnx_files = [x.split('=')[-1] for x in pnnx_args]
    for f_debug in ('debug.bin', 'debug.param', 'debug2.bin',
        'debug2.param', *pnnx_files):
        Path(f_debug).unlink(missing_ok=True)
    yaml_save(f / 'metadata.yaml', self.metadata)
    return str(f), None
