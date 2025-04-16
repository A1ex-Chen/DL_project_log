def contains(seq, item):
    """Jinja2 custom test to check existence in a container.
    Example of use:
    {% set class_methods = methods|selectattr("properties", "contains", "classmethod") %}
    Related doc: https://jinja.palletsprojects.com/en/3.1.x/api/#custom-tests
    """
    return item in seq
