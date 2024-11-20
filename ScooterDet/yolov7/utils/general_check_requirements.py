def check_requirements(requirements='requirements.txt', exclude=()):
    import pkg_resources as pkg
    prefix = colorstr('red', 'bold', 'requirements:')
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        if not file.exists():
            print(f'{prefix} {file.resolve()} not found, check failed.')
            return
        requirements = [f'{x.name}{x.specifier}' for x in pkg.
            parse_requirements(file.open()) if x.name not in exclude]
    else:
        requirements = [x for x in requirements if x not in exclude]
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:
            n += 1
            print(
                f'{prefix} {e.req} not found and is required by YOLOR, attempting auto-update...'
                )
            print(subprocess.check_output(f"pip install '{e.req}'", shell=
                True).decode())
    if n:
        source = file.resolve() if 'file' in locals() else requirements
        s = f"""{prefix} {n} package{'s' * (n > 1)} updated per {source}
{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}
"""
        print(emojis(s))
