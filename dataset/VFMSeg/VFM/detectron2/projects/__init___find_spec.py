def find_spec(self, name, path, target=None):
    if not name.startswith('detectron2.projects.'):
        return
    project_name = name.split('.')[-1]
    project_dir = _PROJECTS.get(project_name)
    if not project_dir:
        return
    target_file = _PROJECT_ROOT / f'{project_dir}/{project_name}/__init__.py'
    if not target_file.is_file():
        return
    return importlib.util.spec_from_file_location(name, target_file)
