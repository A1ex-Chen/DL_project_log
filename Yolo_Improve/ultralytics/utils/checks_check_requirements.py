@TryExcept()
def check_requirements(requirements=ROOT.parent / 'requirements.txt',
    exclude=(), install=True, cmds=''):
    """
    Check if installed dependencies meet YOLOv8 requirements and attempt to auto-update if needed.

    Args:
        requirements (Union[Path, str, List[str]]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (Tuple[str]): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.

    Example:
        ```python
        from ultralytics.utils.checks import check_requirements

        # Check a requirements.txt file
        check_requirements('path/to/requirements.txt')

        # Check a single package
        check_requirements('ultralytics>=8.0.0')

        # Check multiple packages
        check_requirements(['numpy', 'ultralytics>=8.0.0'])
        ```
    """
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()
    check_torchvision()
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f'{prefix} {file} not found, check failed.'
        requirements = [f'{x.name}{x.specifier}' for x in
            parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]
    pkgs = []
    for r in requirements:
        r_stripped = r.split('/')[-1].replace('.git', '')
        match = re.match('([a-zA-Z0-9-_]+)([<>!=~]+.*)?', r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ''
        try:
            assert check_version(metadata.version(name), required)
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands):
        """Attempt pip install command with retries on failure."""
        return subprocess.check_output(
            f'pip install --no-cache-dir {packages} {commands}', shell=True
            ).decode()
    s = ' '.join(f'"{x}"' for x in pkgs)
    if s:
        if install and AUTOINSTALL:
            n = len(pkgs)
            LOGGER.info(
                f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate..."
                )
            try:
                t = time.time()
                assert ONLINE, 'AutoUpdate skipped (offline)'
                LOGGER.info(attempt_install(s, cmds))
                dt = time.time() - t
                LOGGER.info(
                    f"""{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}
{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}
"""
                    )
            except Exception as e:
                LOGGER.warning(f'{prefix} ❌ {e}')
                return False
        else:
            return False
    return True
