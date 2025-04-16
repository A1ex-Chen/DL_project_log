def web_project_name(project):
    if not project.startswith('runs/train'):
        return project
    suffix = '-Classify' if project.endswith('-cls'
        ) else '-Segment' if project.endswith('-seg') else ''
    return f'YOLOv5{suffix}'
