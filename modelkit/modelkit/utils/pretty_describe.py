def describe(obj, t=None):
    if not t:
        t = Tree('')
    if hasattr(obj, 'model_fields'):
        for field_name, field in obj.model_fields.items():
            field_type = field.annotation
            sub_t = t.add(
                f'[deep_sky_blue1]{field_name}[/deep_sky_blue1] [dim]: {pretty_print_type(field_type)}[/dim]'
                )
            describe(getattr(obj, field_name), t=sub_t)
    elif isinstance(obj, dict):
        if obj:
            for field_name, field in obj.items():
                sub_t = t.add(
                    f'[deep_sky_blue1]{escape(field_name)}[/deep_sky_blue1] [dim]: {pretty_print_type(type(field).__name__)}[/dim]'
                    )
                describe(field, t=sub_t)
        else:
            t.label += ' = [orange3]{}[/orange3]'
    elif isinstance(obj, type):
        t.label += f' = [orange3]{escape(str(obj.__name__))}[/orange3] type'
    elif isinstance(obj, (str, int, float, bool)):
        t.label += f' = [orange3]{escape(repr(obj))}[/orange3]'
    elif obj is None:
        t.label += ' = [orange3]None[/orange3]'
    elif isinstance(obj, object):
        t.label += (
            f' = [orange3]{escape(str(obj.__class__.__name__))}[/orange3] instance'
            )
        try:
            obj.describe(t=t)
        except AttributeError:
            try:
                for var in vars(obj):
                    if var.startswith('_'):
                        continue
                    sub_t = t.add(
                        f'[deep_sky_blue1]{var}[/deep_sky_blue1] [dim]: {pretty_print_type(type(getattr(obj, var)).__name__)}[/dim] = {escape(repr(getattr(obj, var)))}'
                        )
            except TypeError:
                t.label += f' = [orange3]{escape(str(obj))}[/orange3]'
    return t
