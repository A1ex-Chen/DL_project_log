def _gen_instance_class(fields):
    """
    Args:
        fields (dict[name: type])
    """


    class _FieldType:

        def __init__(self, name, type_):
            assert isinstance(name, str), f'Field name must be str, got {name}'
            self.name = name
            self.type_ = type_
            self.annotation = f'{type_.__module__}.{type_.__name__}'
    fields = [_FieldType(k, v) for k, v in fields.items()]

    def indent(level, s):
        return ' ' * 4 * level + s
    lines = []
    global _counter
    _counter += 1
    cls_name = 'ScriptedInstances{}'.format(_counter)
    field_names = tuple(x.name for x in fields)
    extra_args = ', '.join([f'{f.name}: Optional[{f.annotation}] = None' for
        f in fields])
    lines.append(
        f"""
class {cls_name}:
    def __init__(self, image_size: Tuple[int, int], {extra_args}):
        self.image_size = image_size
        self._field_names = {field_names}
"""
        )
    for f in fields:
        lines.append(indent(2,
            f'self._{f.name} = torch.jit.annotate(Optional[{f.annotation}], {f.name})'
            ))
    for f in fields:
        lines.append(
            f"""
    @property
    def {f.name}(self) -> {f.annotation}:
        # has to use a local for type refinement
        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        t = self._{f.name}
        assert t is not None, "{f.name} is None and cannot be accessed!"
        return t

    @{f.name}.setter
    def {f.name}(self, value: {f.annotation}) -> None:
        self._{f.name} = value
"""
            )
    lines.append("""
    def __len__(self) -> int:
""")
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            return len(t)
"""
            )
    lines.append(
        """
        raise NotImplementedError("Empty Instances does not support __len__!")
"""
        )
    lines.append("""
    def has(self, name: str) -> bool:
""")
    for f in fields:
        lines.append(
            f"""
        if name == "{f.name}":
            return self._{f.name} is not None
"""
            )
    lines.append("""
        return False
""")
    none_args = ', None' * len(fields)
    lines.append(
        f"""
    def to(self, device: torch.device) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
        )
    for f in fields:
        if hasattr(f.type_, 'to'):
            lines.append(
                f"""
        t = self._{f.name}
        if t is not None:
            ret._{f.name} = t.to(device)
"""
                )
        else:
            pass
    lines.append("""
        return ret
""")
    none_args = ', None' * len(fields)
    lines.append(
        f"""
    def __getitem__(self, item) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
        )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            ret._{f.name} = t[item]
"""
            )
    lines.append("""
        return ret
""")
    none_args = ', None' * len(fields)
    lines.append(
        f"""
    def cat(self, instances: List["{cls_name}"]) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
        )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            values: List[{f.annotation}] = [x.{f.name} for x in instances]
            if torch.jit.isinstance(t, torch.Tensor):
                ret._{f.name} = torch.cat(values, dim=0)
            else:
                ret._{f.name} = t.cat(values)
"""
            )
    lines.append("""
        return ret""")
    lines.append(
        """
    def get_fields(self) -> Dict[str, Tensor]:
        ret = {}
    """
        )
    for f in fields:
        if f.type_ == Boxes:
            stmt = 't.tensor'
        elif f.type_ == torch.Tensor:
            stmt = 't'
        else:
            stmt = f'assert False, "unsupported type {str(f.type_)}"'
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            ret["{f.name}"] = {stmt}
        """
            )
    lines.append("""
        return ret""")
    return cls_name, os.linesep.join(lines)
