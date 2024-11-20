@TryExcept()
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(),
    install=True, cmds=''):
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f'{prefix} {file} not found, check failed.'
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.
                parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]
    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):
            s += f'"{r}" '
            n += 1
    if s and install and AUTOINSTALL:
        LOGGER.info(
            f"{prefix} YOLOv5 requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate..."
            )
        try:
            LOGGER.info(check_output(f'pip install {s} {cmds}', shell=True)
                .decode())
            source = file if 'file' in locals() else requirements
            s = f"""{prefix} {n} package{'s' * (n > 1)} updated per {source}
{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}
"""
            LOGGER.info(s)
        except Exception as e:
            LOGGER.warning(f'{prefix} ❌ {e}')
