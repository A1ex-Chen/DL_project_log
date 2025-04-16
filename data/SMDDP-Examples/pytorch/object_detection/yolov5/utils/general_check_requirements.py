@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(),
    install=True, cmds=()):
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        assert file.exists(
            ), f'{prefix} {file.resolve()} not found, check failed.'
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.
                parse_requirements(f) if x.name not in exclude]
    else:
        requirements = [x for x in requirements if x not in exclude]
    n = 0
    for i, r in enumerate(requirements):
        try:
            pkg.require(r)
        except Exception:
            s = f'{prefix} {r} not found and is required by YOLOv5'
            if install and AUTOINSTALL:
                LOGGER.info(f'{s}, attempting auto-update...')
                try:
                    assert check_online(
                        ), f"'pip install {r}' skipped (offline)"
                    LOGGER.info(check_output(
                        f'pip install "{r}" {cmds[i] if cmds else \'\'}',
                        shell=True).decode())
                    n += 1
                except Exception as e:
                    LOGGER.warning(f'{prefix} {e}')
            else:
                LOGGER.info(f'{s}. Please install and rerun your command.')
    if n:
        source = file.resolve() if 'file' in locals() else requirements
        s = f"""{prefix} {n} package{'s' * (n > 1)} updated per {source}
{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}
"""
        LOGGER.info(s)
