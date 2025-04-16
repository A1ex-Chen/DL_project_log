@staticmethod
def scan_for_extensions(extension_dirs: List[Path]):
    register_pattern = '.*\\.register_extension\\(.*'
    for extension_dir in extension_dirs:
        for python_path in extension_dir.rglob('*.py'):
            if not python_path.is_file():
                continue
            payload = python_path.read_text()
            if re.findall(register_pattern, payload):
                import_path = python_path.relative_to(toolkit_root_dir.parent)
                package = import_path.parent.as_posix().replace(os.sep, '.')
                package_with_module = f'{package}.{import_path.stem}'
                spec = importlib.util.spec_from_file_location(name=
                    package_with_module, location=python_path)
                my_module = importlib.util.module_from_spec(spec)
                my_module.__package__ = package
                try:
                    spec.loader.exec_module(my_module)
                except ModuleNotFoundError as e:
                    LOGGER.error(
                        f'Could not load extensions from {import_path} due to missing python packages; {e}'
                        )
