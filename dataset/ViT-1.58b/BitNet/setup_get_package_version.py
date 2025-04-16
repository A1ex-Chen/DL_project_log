def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / '__init__.py', 'r') as f:
        version_match = re.search('^__version__\\s*=\\s*(.*)$', f.read(),
            re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get('MAMBA_LOCAL_VERSION')
    if local_version:
        return f'{public_version}+{local_version}'
    else:
        return str(public_version)
