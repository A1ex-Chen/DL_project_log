def _add_dataclass_arguments(self, dtype: DataClassType):
    for field in dataclasses.fields(dtype):
        field_name = f'--{field.name}'
        kwargs = field.metadata.copy()
        if isinstance(field.type, str):
            raise ImportError(
                'This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),which can be opted in from Python 3.7 with `from __future__ import annotations`.We will add compatibility when Python 3.9 is released.'
                )
        typestring = str(field.type)
        for prim_type in (int, float, str):
            for collection in (List,):
                if (typestring ==
                    f'typing.Union[{collection[prim_type]}, NoneType]'):
                    field.type = collection[prim_type]
            if typestring == f'typing.Union[{prim_type.__name__}, NoneType]':
                field.type = prim_type
        if isinstance(field.type, type) and issubclass(field.type, Enum):
            kwargs['choices'] = list(field.type)
            kwargs['type'] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs['default'] = field.default
        elif field.type is bool or field.type is Optional[bool]:
            if (field.type is bool or field.default is not None and field.
                default is not dataclasses.MISSING):
                kwargs['action'
                    ] = 'store_false' if field.default is True else 'store_true'
            if field.default is True:
                field_name = f'--no_{field.name}'
                kwargs['dest'] = field.name
        elif hasattr(field.type, '__origin__') and issubclass(field.type.
            __origin__, List):
            kwargs['nargs'] = '+'
            kwargs['type'] = field.type.__args__[0]
            assert all(x == kwargs['type'] for x in field.type.__args__
                ), '{} cannot be a List of mixed types'.format(field.name)
            if field.default_factory is not dataclasses.MISSING:
                kwargs['default'] = field.default_factory()
        else:
            kwargs['type'] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs['default'] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs['default'] = field.default_factory()
            else:
                kwargs['required'] = True
        self.add_argument(field_name, **kwargs)
