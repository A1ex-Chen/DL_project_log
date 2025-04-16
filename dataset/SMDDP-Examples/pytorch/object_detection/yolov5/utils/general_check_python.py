def check_python(minimum='3.7.0'):
    check_version(platform.python_version(), minimum, name='Python ', hard=True
        )
